uniform float4x4 model;
uniform float4x4 view;
uniform float4x4 proj;
uniform float3 cameraPos;
uniform Texture2D<float4> texture;
uniform SamplerState sampler;

struct VertexInput {
    float3 pos : POSITION;
    float3 norm : NORMAL;
    float2 uv : TEXCOORD;
};

struct VertexOutput {
    float4 pos : SV_Position;
    float3 worldPos : TEXCOORD1;
    float3 normal : TEXCOORD2;
    float3 flatNormal : TEXCOORD3;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput in) {
    float4x4 mv = mul(view, model);

    VertexOutput out = (VertexOutput)0;
    float4 worldPosition = mul(model, float4(in.pos, 1.0));
    out.pos = mul(proj, mul(view, worldPosition));
    out.worldPos = worldPosition.xyz;

    float3 worldNormal = normalize(mul((float3x3)model, in.norm));
    out.normal = worldNormal;

    out.uv = in.uv;
    return out;
}

float3 ComputeFlatNormal(float3 pos0, float3 pos1, float3 pos2, bool isOddNegative)
{
    float3 flatNormal = normalize(cross(pos1 - pos0, pos2 - pos0));

    if (isOddNegative)
    {
        flatNormal = -flatNormal;
    }

    return flatNormal;
}

[shader("geometry")]
[maxvertexcount(3)]
void geometry_main(triangle VertexOutput Input[3], inout TriangleStream<VertexOutput> Output)
{
    float3 flatNormal = ComputeFlatNormal(Input[0].worldPos, Input[1].worldPos, Input[2].worldPos, false);

    VertexOutput outputVertexA = Input[0];
    outputVertexA.flatNormal = flatNormal;
    VertexOutput outputVertexB = Input[1];
    outputVertexB.flatNormal = flatNormal;
    VertexOutput outputVertexC = Input[2];
    outputVertexC.flatNormal = flatNormal;

    Output.Append(outputVertexA);
    Output.Append(outputVertexB);
    Output.Append(outputVertexC);
}

[shader("fragment")]
float4 fragment_main(VertexOutput in) : SV_Target {
    float3 lightPos = { -2, 5, 5 };

    float3 N = normalize(in.flatNormal);
    float3 L = normalize(lightPos - in.worldPos);
    float3 V = normalize(cameraPos - in.worldPos);
    float3 R = reflect(-L, N);

    float ambientStrength = 0.1;
    float3 ambient = ambientStrength * float3(1.0, 1.0, 1.0);

    float diff = max(dot(N, L), 0.0);
    float3 diffuse = diff * float3(1.0, 1.0, 1.0);

    float specularStrength = 0.5;
    float shininess = 32.0;
    float spec = pow(max(dot(R, V), 0.0), shininess);
    float3 specular = specularStrength * spec * float3(1.0, 1.0, 1.0);

    float3 lighting = ambient + diffuse + specular;

    float4 texColor = texture.Sample(sampler, in.uv);
    return float4(texColor.rgb * lighting, texColor.a);
}
