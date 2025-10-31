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

const float PI = 3.14159265359;

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float distributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / max(PI * denom * denom, 1e-5);
}

float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float denom = NdotV * (1.0 - k) + k;
    return NdotV / max(denom, 1e-5);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

void main()
{
    vec3 albedo = texture(albedoMap, fs_in.TexCoords).rgb;
    vec3 normalSample = texture(normalMap, fs_in.TexCoords).rgb;
    normalSample = normalSample * 2.0 - 1.0;
    vec3 N = normalize(fs_in.TBN * normalSample);

    float metallic = texture(metallicMap, fs_in.TexCoords).r;
    float roughness = clamp(texture(roughnessMap, fs_in.TexCoords).r, 0.05, 1.0);
    float ao = texture(aoMap, fs_in.TexCoords).r;

    vec3 V = normalize(camPos - fs_in.WorldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 Lo = vec3(0.0);
    for (int i = 0; i < 4; ++i)
    {
        vec3 L = lightPositions[i] - fs_in.WorldPos;
        float distance = length(L);
        L = normalize(L);
        vec3 H = normalize(V + L);
        vec3 radiance = lightColors[i] * (lightIntensities[i] / max(distance * distance, 1e-4));

        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
        float VdotH = max(dot(V, H), 0.0);

        float D = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(VdotH, F0);

        vec3 numerator = D * G * F;
        float denominator = max(4.0 * NdotV * NdotL, 1e-5);
        vec3 specular = numerator / denominator;

        vec3 kS = F;
        vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
        vec3 diffuse = kD * albedo / PI;

        Lo += (diffuse + specular) * radiance * NdotL;
    }

    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
