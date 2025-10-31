import json
import math
import random
from pathlib import Path
from typing import List, Tuple

PI = math.pi


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def normalize(v: List[float]) -> List[float]:
    length = math.sqrt(max(dot(v, v), 1e-12))
    return [x / length for x in v]


def random_hemisphere_direction() -> List[float]:
    u1 = random.random()
    u2 = random.random()
    theta = math.acos(1.0 - u1)
    phi = 2.0 * PI * u2
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    return [x, y, z]


def fresnel_schlick(cos_theta: float, f0: List[float]) -> List[float]:
    factor = (1.0 - cos_theta)
    factor = max(min(factor, 1.0), 0.0)
    factor = factor ** 5.0
    return [f0_i + (1.0 - f0_i) * factor for f0_i in f0]


def distribution_ggx(n_dot_h: float, roughness: float) -> float:
    a = roughness * roughness
    a2 = a * a
    n_dot_h2 = n_dot_h * n_dot_h
    denom = n_dot_h2 * (a2 - 1.0) + 1.0
    return a2 / max(PI * denom * denom, 1e-6)


def geometry_schlick_ggx(n_dot_v: float, roughness: float) -> float:
    r = roughness + 1.0
    k = (r * r) / 8.0
    denom = n_dot_v * (1.0 - k) + k
    return n_dot_v / max(denom, 1e-6)


def geometry_smith(n_dot_v: float, n_dot_l: float, roughness: float) -> float:
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness)


def ue4_brdf(base_color: List[float], roughness: float, metallic: float,
             n: List[float], v: List[float], l: List[float]) -> Tuple[List[float], float, float, float, float]:
    v = normalize(v)
    l = normalize(l)
    h = normalize([v[i] + l[i] for i in range(3)])

    n_dot_v = max(dot(n, v), 0.0)
    n_dot_l = max(dot(n, l), 0.0)
    n_dot_h = max(dot(n, h), 0.0)
    v_dot_h = max(dot(v, h), 0.0)

    f0 = [0.04 * (1.0 - metallic) + base_color[i] * metallic for i in range(3)]
    fresnel = fresnel_schlick(v_dot_h, f0)
    distribution = distribution_ggx(n_dot_h, roughness)
    geometry = geometry_smith(n_dot_v, n_dot_l, roughness)

    spec_num = [distribution * geometry * fresnel_i for fresnel_i in fresnel]
    spec_denom = 4.0 * max(n_dot_v * n_dot_l, 1e-6)
    specular = [s / spec_denom for s in spec_num]

    k_s = fresnel
    k_d = [(1.0 - k_s[i]) * (1.0 - metallic) for i in range(3)]
    diffuse = [k_d[i] * base_color[i] / PI for i in range(3)]

    color = [(diffuse[i] + specular[i]) * n_dot_l for i in range(3)]
    return color, n_dot_v, n_dot_l, n_dot_h, v_dot_h


def sample_dataset(num_samples: int) -> Tuple[List[List[float]], List[List[float]]]:
    normal = [0.0, 0.0, 1.0]
    features: List[List[float]] = []
    targets: List[List[float]] = []
    for _ in range(num_samples):
        base_color = [random.random() ** 2.2 for _ in range(3)]
        roughness = max(min(random.betavariate(2.0, 5.0), 1.0), 0.05)
        metallic = random.random()
        v = random_hemisphere_direction()
        l = random_hemisphere_direction()
        radiance, n_dot_v, n_dot_l, _, v_dot_h = ue4_brdf(base_color, roughness, metallic, normal, v, l)
        feat = [
            base_color[0],
            base_color[1],
            base_color[2],
            roughness,
            metallic,
            n_dot_v,
            n_dot_l,
            v_dot_h,
        ]
        features.append(feat)
        targets.append(radiance)
    return features, targets


def vector_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def vector_scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]


def relu(v: List[float]) -> List[float]:
    return [x if x > 0.0 else 0.0 for x in v]


def relu_derivative(v: List[float]) -> List[float]:
    return [1.0 if x > 0.0 else 0.0 for x in v]


def matvec(W: List[List[float]], x: List[float]) -> List[float]:
    return [sum(w_ij * x_j for w_ij, x_j in zip(row, x)) for row in W]


def matvec_batch(W: List[List[float]], xs: List[List[float]]) -> List[List[float]]:
    return [matvec(W, x) for x in xs]


def initialize_matrix(rows: int, cols: int, scale: float) -> List[List[float]]:
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def zero_matrix(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def zero_vector(size: int) -> List[float]:
    return [0.0 for _ in range(size)]


def train_mlp(features: List[List[float]], targets: List[List[float]],
              epochs: int = 200, batch_size: int = 32, lr: float = 1e-3) -> Tuple[dict, float, float]:
    input_dim = len(features[0])
    hidden_dim = 32
    hidden_dim2 = 32
    output_dim = len(targets[0])

    W1 = initialize_matrix(hidden_dim, input_dim, math.sqrt(2.0 / input_dim))
    b1 = zero_vector(hidden_dim)
    W2 = initialize_matrix(hidden_dim2, hidden_dim, math.sqrt(2.0 / hidden_dim))
    b2 = zero_vector(hidden_dim2)
    W3 = initialize_matrix(output_dim, hidden_dim2, math.sqrt(2.0 / hidden_dim2))
    b3 = zero_vector(output_dim)

    num_samples = len(features)

    # compute mean/std for normalization
    mean = [0.0 for _ in range(input_dim)]
    for feat in features:
        for i, x in enumerate(feat):
            mean[i] += x
    mean = [m / num_samples for m in mean]

    std = [0.0 for _ in range(input_dim)]
    for feat in features:
        for i, x in enumerate(feat):
            diff = x - mean[i]
            std[i] += diff * diff
    std = [math.sqrt(s / max(num_samples - 1, 1)) for s in std]
    std = [s if s > 1e-6 else 1.0 for s in std]

    def normalize_feature(feat: List[float]) -> List[float]:
        return [(feat[i] - mean[i]) / std[i] for i in range(input_dim)]

    norm_features = [normalize_feature(f) for f in features]

    for epoch in range(epochs):
        combined = list(zip(norm_features, targets))
        random.shuffle(combined)
        norm_features[:], targets[:] = zip(*combined)
        total_loss = 0.0
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_inputs = norm_features[start:end]
            batch_targets = targets[start:end]

            z1 = [vector_add(matvec(W1, x), b1) for x in batch_inputs]
            a1 = [relu(v) for v in z1]
            z2 = [vector_add(matvec(W2, v), b2) for v in a1]
            a2 = [relu(v) for v in z2]
            outputs = [vector_add(matvec(W3, v), b3) for v in a2]

            batch_loss = 0.0
            for out, tgt in zip(outputs, batch_targets):
                batch_loss += sum((o - t) ** 2 for o, t in zip(out, tgt))
            batch_loss /= (end - start)
            total_loss += batch_loss * (end - start)

            grad_W3 = zero_matrix(output_dim, hidden_dim2)
            grad_b3 = zero_vector(output_dim)
            grad_W2 = zero_matrix(hidden_dim2, hidden_dim)
            grad_b2 = zero_vector(hidden_dim2)
            grad_W1 = zero_matrix(hidden_dim, input_dim)
            grad_b1 = zero_vector(hidden_dim)

            for x, tgt, z1_v, a1_v, z2_v, a2_v, out in zip(batch_inputs, batch_targets, z1, a1, z2, a2, outputs):
                diff = [2.0 * (out[i] - tgt[i]) / (end - start) for i in range(output_dim)]

                for i in range(output_dim):
                    grad_b3[i] += diff[i]
                    for j in range(hidden_dim2):
                        grad_W3[i][j] += diff[i] * a2_v[j]

                grad_a2 = [sum(diff[i] * W3[i][j] for i in range(output_dim)) for j in range(hidden_dim2)]
                grad_z2 = [grad_a2[j] * (1.0 if z2_v[j] > 0.0 else 0.0) for j in range(hidden_dim2)]

                for i in range(hidden_dim2):
                    grad_b2[i] += grad_z2[i]
                    for j in range(hidden_dim):
                        grad_W2[i][j] += grad_z2[i] * a1_v[j]

                grad_a1 = [sum(grad_z2[i] * W2[i][j] for i in range(hidden_dim2)) for j in range(hidden_dim)]
                grad_z1 = [grad_a1[j] * (1.0 if z1_v[j] > 0.0 else 0.0) for j in range(hidden_dim)]

                for i in range(hidden_dim):
                    grad_b1[i] += grad_z1[i]
                    for j in range(input_dim):
                        grad_W1[i][j] += grad_z1[i] * x[j]

            for i in range(output_dim):
                b3[i] -= lr * grad_b3[i]
                for j in range(hidden_dim2):
                    W3[i][j] -= lr * grad_W3[i][j]

            for i in range(hidden_dim2):
                b2[i] -= lr * grad_b2[i]
                for j in range(hidden_dim):
                    W2[i][j] -= lr * grad_W2[i][j]

            for i in range(hidden_dim):
                b1[i] -= lr * grad_b1[i]
                for j in range(input_dim):
                    W1[i][j] -= lr * grad_W1[i][j]

        avg_loss = total_loss / num_samples
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    weights = {
        "input_mean": mean,
        "input_std": std,
        "layers": [
            {"weights": W1, "bias": b1},
            {"weights": W2, "bias": b2},
            {"weights": W3, "bias": b3},
        ],
        "architecture": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "hidden_dim2": hidden_dim2,
            "output_dim": output_dim,
        },
    }
    return weights


def main():
    random.seed(42)
    features, targets = sample_dataset(4000)
    weights = train_mlp(features, targets, epochs=120, batch_size=32, lr=1e-3)

    output_path = Path("LearnOpenGL/LearnOpenGL/assets/mlp/mlp_weights.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    print(f"Saved weights to {output_path}")


if __name__ == "__main__":
    main()
