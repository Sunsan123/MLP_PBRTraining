import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

PI = np.pi


def fresnel_schlick(cos_theta: np.ndarray, f0: np.ndarray) -> np.ndarray:
    return f0 + (1.0 - f0) * np.power(np.clip(1.0 - cos_theta, 0.0, 1.0), 5.0)


def distribution_ggx(n_dot_h: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    a = roughness ** 2.0
    a2 = a * a
    n_dot_h2 = n_dot_h * n_dot_h
    denom = n_dot_h2 * (a2 - 1.0) + 1.0
    return a2 / (PI * denom * denom + 1e-6)


def geometry_schlick_ggx(n_dot_v: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    r = (roughness + 1.0)
    k = (r * r) / 8.0
    denom = n_dot_v * (1.0 - k) + k
    return n_dot_v / (denom + 1e-6)


def geometry_smith(n_dot_v: np.ndarray, n_dot_l: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def random_hemisphere_directions(num: int) -> np.ndarray:
    u1 = np.random.rand(num)
    u2 = np.random.rand(num)
    theta = np.arccos(1.0 - u1)
    phi = 2.0 * PI * u2
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    dirs = np.stack([x, y, z], axis=1)
    return dirs


def ue4_brdf(base_color: np.ndarray, roughness: np.ndarray, metallic: np.ndarray,
             n: np.ndarray, v: np.ndarray, l: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = normalize(v)
    l = normalize(l)
    h = normalize(v + l)

    n_dot_v = np.clip(np.sum(n * v, axis=1), 0.0, 1.0)
    n_dot_l = np.clip(np.sum(n * l, axis=1), 0.0, 1.0)
    n_dot_h = np.clip(np.sum(n * h, axis=1), 0.0, 1.0)
    v_dot_h = np.clip(np.sum(v * h, axis=1), 0.0, 1.0)

    # Base reflectivity at normal incidence
    f0 = 0.04 * (1.0 - metallic) + base_color * metallic

    fresnel = fresnel_schlick(v_dot_h[:, None], f0)
    distribution = distribution_ggx(n_dot_h, roughness)
    geometry = geometry_smith(n_dot_v, n_dot_l, roughness)

    spec_num = distribution[:, None] * geometry[:, None] * fresnel
    spec_denom = 4.0 * n_dot_v * n_dot_l
    specular = spec_num / (spec_denom[:, None] + 1e-6)

    k_s = fresnel
    k_d = (1.0 - k_s) * (1.0 - metallic)
    diffuse = (k_d * base_color) / PI

    color = (diffuse + specular) * n_dot_l[:, None]
    return color, n_dot_v, n_dot_l, n_dot_h, v_dot_h


def sample_dataset(num_samples: int, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    base_color = np.random.rand(num_samples, 3) ** 2.2  # bias towards darker colors
    roughness = np.clip(np.random.beta(2.0, 5.0, size=(num_samples, 1)), 0.05, 1.0)
    metallic = np.random.rand(num_samples, 1)

    v = random_hemisphere_directions(num_samples)
    l = random_hemisphere_directions(num_samples)
    n = np.repeat(normal[None, :], num_samples, axis=0)

    radiance, n_dot_v, n_dot_l, n_dot_h, v_dot_h = ue4_brdf(base_color, roughness, metallic, n, v, l)

    features = np.concatenate([
        base_color,
        roughness,
        metallic,
        n_dot_v[:, None],
        n_dot_l[:, None],
        v_dot_h[:, None]
    ], axis=1)
    return features.astype(np.float32), radiance.astype(np.float32)


class AdamOptimizer:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, hidden_dim2: int, output_dim: int):
        rng = np.random.default_rng()
        self.W1 = rng.standard_normal((hidden_dim, input_dim)) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.W2 = rng.standard_normal((hidden_dim2, hidden_dim)) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((hidden_dim2,), dtype=np.float32)
        self.W3 = rng.standard_normal((output_dim, hidden_dim2)) * np.sqrt(2.0 / hidden_dim2)
        self.b3 = np.zeros((output_dim,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z1 = x @ self.W1.T + self.b1
        self.a1 = np.maximum(self.z1, 0.0)
        self.z2 = self.a1 @ self.W2.T + self.b2
        self.a2 = np.maximum(self.z2, 0.0)
        self.z3 = self.a2 @ self.W3.T + self.b3
        return self.z3

    def backward(self, grad_output: np.ndarray):
        grad_z3 = grad_output
        grad_W3 = grad_z3.T @ self.a2 / grad_output.shape[0]
        grad_b3 = grad_z3.mean(axis=0)

        grad_a2 = grad_z3 @ self.W3
        grad_z2 = grad_a2 * (self.z2 > 0.0)
        grad_W2 = grad_z2.T @ self.a1 / grad_output.shape[0]
        grad_b2 = grad_z2.mean(axis=0)

        grad_a1 = grad_z2 @ self.W2
        grad_z1 = grad_a1 * (self.z1 > 0.0)
        grad_W1 = grad_z1.T @ self.x / grad_output.shape[0]
        grad_b1 = grad_z1.mean(axis=0)

        grad_input = grad_z1 @ self.W1
        return (grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3), grad_input

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]


def train(dataset: Tuple[np.ndarray, np.ndarray], epochs: int, batch_size: int, lr: float):
    inputs, targets = dataset
    num_samples, feature_dim = inputs.shape
    output_dim = targets.shape[1]
    hidden_dim = 64
    hidden_dim2 = 64
    model = MLP(feature_dim, hidden_dim, hidden_dim2, output_dim)
    optimizer = AdamOptimizer(model.parameters(), lr=lr)

    for epoch in range(epochs):
        perm = np.random.permutation(num_samples)
        inputs = inputs[perm]
        targets = targets[perm]
        total_loss = 0.0
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            x_batch = inputs[start:end]
            y_batch = targets[start:end]

            preds = model.forward(x_batch)
            diff = preds - y_batch
            loss = np.mean(diff * diff)
            total_loss += loss * (end - start)

            grad_output = 2.0 * diff / (end - start)
            grads, _ = model.backward(grad_output)
            optimizer.step(grads)

        avg_loss = total_loss / num_samples
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train an MLP to approximate UE4 BRDF shading")
    parser.add_argument("--samples", type=int, default=200000, help="Number of BRDF samples to generate")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=Path, default=Path("LearnOpenGL/LearnOpenGL/assets/mlp/mlp_weights.json"),
                        help="Output path for serialized network weights")
    parser.add_argument("--dataset", type=Path, default=Path("dataset_brdf.npz"), help="Optional output dataset path")
    args = parser.parse_args()

    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    features, radiance = sample_dataset(args.samples, normal)

    np.savez(args.dataset, features=features, radiance=radiance)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    standardized_features = (features - feature_mean) / feature_std

    model = train((standardized_features, radiance), epochs=args.epochs, batch_size=args.batch, lr=args.lr)

    weights = {
        "input_mean": feature_mean.tolist(),
        "input_std": feature_std.tolist(),
        "layers": [
            {"weights": model.W1.tolist(), "bias": model.b1.tolist()},
            {"weights": model.W2.tolist(), "bias": model.b2.tolist()},
            {"weights": model.W3.tolist(), "bias": model.b3.tolist()},
        ],
        "architecture": {
            "input_dim": int(features.shape[1]),
            "hidden_dim": int(model.W1.shape[0]),
            "hidden_dim2": int(model.W2.shape[0]),
            "output_dim": int(model.W3.shape[0])
        }
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    print(f"Saved weights to {args.output}")
    print(f"Saved dataset to {args.dataset}")


if __name__ == "__main__":
    main()
