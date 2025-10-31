#version 330 core
out vec4 FragColor;

in VS_OUT
{
    vec3 WorldPos;
    vec2 TexCoords;
    mat3 TBN;
} fs_in;

uniform vec3 camPos;
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;

uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];
uniform float lightIntensities[4];

#define INPUT_DIM 8
#define HIDDEN0_DIM 32
#define HIDDEN1_DIM 32
#define OUTPUT_DIM 3

uniform float uInputMean[INPUT_DIM];
uniform float uInputInvStd[INPUT_DIM];
uniform float uLayer0Weights[HIDDEN0_DIM * INPUT_DIM];
uniform float uLayer0Bias[HIDDEN0_DIM];
uniform float uLayer1Weights[HIDDEN1_DIM * HIDDEN0_DIM];
uniform float uLayer1Bias[HIDDEN1_DIM];
uniform float uLayer2Weights[OUTPUT_DIM * HIDDEN1_DIM];
uniform float uLayer2Bias[OUTPUT_DIM];

const float PI = 3.14159265359;

vec3 evaluateMlp(vec3 albedo, float roughness, float metallic, float NdotV, float NdotL, float VdotH)
{
    float input[INPUT_DIM];
    input[0] = albedo.r;
    input[1] = albedo.g;
    input[2] = albedo.b;
    input[3] = roughness;
    input[4] = metallic;
    input[5] = NdotV;
    input[6] = NdotL;
    input[7] = VdotH;

    float normalized[INPUT_DIM];
    for (int i = 0; i < INPUT_DIM; ++i)
    {
        normalized[i] = (input[i] - uInputMean[i]) * uInputInvStd[i];
    }

    float hidden0[HIDDEN0_DIM];
    for (int i = 0; i < HIDDEN0_DIM; ++i)
    {
        float sum = uLayer0Bias[i];
        for (int j = 0; j < INPUT_DIM; ++j)
        {
            sum += uLayer0Weights[i * INPUT_DIM + j] * normalized[j];
        }
        hidden0[i] = max(sum, 0.0);
    }

    float hidden1[HIDDEN1_DIM];
    for (int i = 0; i < HIDDEN1_DIM; ++i)
    {
        float sum = uLayer1Bias[i];
        for (int j = 0; j < HIDDEN0_DIM; ++j)
        {
            sum += uLayer1Weights[i * HIDDEN0_DIM + j] * hidden0[j];
        }
        hidden1[i] = max(sum, 0.0);
    }

    vec3 output = vec3(0.0);
    for (int i = 0; i < OUTPUT_DIM; ++i)
    {
        float sum = uLayer2Bias[i];
        for (int j = 0; j < HIDDEN1_DIM; ++j)
        {
            sum += uLayer2Weights[i * HIDDEN1_DIM + j] * hidden1[j];
        }
        output[i] = max(sum, 0.0);
    }
    return output;
}

void main()
{
    vec3 albedo = texture(albedoMap, fs_in.TexCoords).rgb;
    vec3 normalSample = texture(normalMap, fs_in.TexCoords).rgb * 2.0 - 1.0;
    vec3 N = normalize(fs_in.TBN * normalSample);
    vec3 V = normalize(camPos - fs_in.WorldPos);

    float metallic = texture(metallicMap, fs_in.TexCoords).r;
    float roughness = clamp(texture(roughnessMap, fs_in.TexCoords).r, 0.05, 1.0);
    float ao = texture(aoMap, fs_in.TexCoords).r;

    float NdotV = max(dot(N, V), 0.0);

    vec3 Lo = vec3(0.0);
    for (int i = 0; i < 4; ++i)
    {
        vec3 L = lightPositions[i] - fs_in.WorldPos;
        float distance = length(L);
        L = normalize(L);
        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float VdotH = max(dot(V, H), 0.0);

        vec3 radiance = lightColors[i] * (lightIntensities[i] / max(distance * distance, 1e-4));
        vec3 brdf = evaluateMlp(albedo, roughness, metallic, NdotV, NdotL, VdotH);
        Lo += brdf * radiance;
    }

    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
