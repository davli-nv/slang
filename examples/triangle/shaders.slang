// shaders.slang

//
// This file provides a simple vertex and fragment shader that can be compiled
// using Slang. This code should also be valid as HLSL, and thus it does not
// use any of the new language features supported by Slang.
//

// Uniform data to be passed from application -> shader.
cbuffer Uniforms
{
    float4x4 modelViewProjection;
}

// Per-vertex attributes to be assembled from bound vertex buffers.
struct AssembledVertex
{
    float3	position : POSITION;
    float3	color    : COLOR;
};

// Output of the vertex shader, and input to the fragment shader.
struct CoarseVertex
{
    float3 color;
};

// Output of the fragment shader
struct Fragment
{
    float4 color;
};

// Vertex  Shader

struct VertexStageOutput
{
    //CoarseVertex    coarseVertex    : CoarseVertex;
    float4          sv_position     : SV_Position;
};

[shader("vertex")]
float4 vertexMain(
    float3 position : POSITION) : SV_Position
{
    VertexStageOutput output;

    //float3 position = position;
    //float3 color    = assembledVertex.color;

    //output.coarseVertex.color = color;
    //output.sv_position = mul(modelViewProjection, float4(position, 1.0));

    //return output;
    return float4(position, 1);
}

// Fragment Shader

struct FragIn
{
    float2 samplePosition : SV_SamplePosition;
};

[shader("fragment")]
float4 fragmentMain(
    //CoarseVertex coarseVertex : CoarseVertex,
    //float4 pixelPosition : SV_Position,
    //uint sampleIndex : SV_SampleIndex
    float2 samplePosition : SV_SamplePosition
    //FragIn input
    ) : SV_Target
{
    //float3 color = coarseVertex.color;

    // Fragment output;
    // output.color = float4(color, 1.0);
    // float4 color = float4(pixelPosition.xy/float2(1024,768), 0, 1.0);
     //float4 color = GetRenderTargetSamplePosition(sampleIndex).xyxy + float4(0.5, 0.5, 0, 1);
     float4 color = float4(samplePosition + float2(0.5, 0.5), 0, 1);
    //float4 color = float4(input.samplePosition + float2(0.5, 0.5), 0, 1);
    return color;
}
