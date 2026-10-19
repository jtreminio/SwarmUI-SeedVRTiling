using SeedVRTiling.Generated;
using ComfyTyped.Core;
using ComfyTyped.Generated;
using Newtonsoft.Json.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using Xunit;

[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace SeedVRTiling.Tests;

public class WorkflowTests
{
    static WorkflowTests()
    {
        ComfyTyped.Generated.NodeRegistrations.EnsureRegistered();
        TileNodeRegistrations.EnsureRegistered();
        ComfyUIBackendExtension.GroupSeedVR = new("SeedVR", Toggles: true);
        ComfyUIBackendExtension.SeedVRModel = T2IParamTypes.Register<T2IModel>(new("SeedVR Model", "", "None"));
        ComfyUIBackendExtension.SeedVRSplitLatent = T2IParamTypes.Register<bool>(new("SeedVR Split Latent", "", "false"));
        T2IParamTypes.ImageFormat = T2IParamTypes.Register<string>(new("Image Format", "", "PNG"));
        SeedVRTilingExtension.RegisterParameters();
    }

    internal static void EnsureInitialized() { }

    private static WorkflowGenerator Generator()
    {
        WorkflowGenerator generator = new()
        {
            Workflow = new JObject(),
            UserInput = new T2IParamInput(null)
        };
        generator.UserInput.Set(SeedVRTilingExtension.Enabled, true);
        generator.UserInput.Set(ComfyUIBackendExtension.SeedVRModel, new T2IModel(null, "", "seedvr.safetensors", "seedvr"));
        return generator;
    }

    private static void NativeGraph(WorkflowGenerator generator)
    {
        using WorkflowBridge bridge = WorkflowBridge.Create(generator.Workflow);
        LoadImageNode image = bridge.AddNode(new LoadImageNode().With(Image: "input.png"), "1");
        SeedVR2PreprocessNode pre = bridge.AddNode(new SeedVR2PreprocessNode().With(ResizedImages: image.IMAGE), "2");
        VAELoaderNode vae = bridge.AddNode(new VAELoaderNode().With(VaeName: "seedvr.safetensors"), "3");
        VAEEncodeNode encode = bridge.AddNode(new VAEEncodeNode().With(Pixels: pre.Images, Vae: vae.VAE), "4");
        UNETLoaderNode model = bridge.AddNode(new UNETLoaderNode().With(UnetName: "seedvr.safetensors"), "5");
        SeedVR2ConditioningNode conditioning = bridge.AddNode(new SeedVR2ConditioningNode().With(Model: model.MODEL, VaeConditioning: encode.LATENT), "6");
        KSamplerNode sample = bridge.AddNode(new KSamplerNode().With(Model: model.MODEL, Positive: conditioning.Positive,
            Negative: conditioning.Negative, LatentImage: encode.LATENT, Seed: 42L, Steps: 1L, Cfg: 1.0,
            SamplerName: "euler", Scheduler: "simple", Denoise: 1.0), "7");
        VAEDecodeNode decode = bridge.AddNode(new VAEDecodeNode().With(Samples: sample.LATENT, Vae: vae.VAE), "8");
        SeedVR2PostProcessingNode post = bridge.AddNode(new SeedVR2PostProcessingNode().With(
            ImagesInput: decode.IMAGE, OriginalResizedImages: image.IMAGE, ColorCorrectionMethod: "wavelet"), "9");
        bridge.AddNode(new SaveImageNode().With(ImagesInput: post.Images, FilenamePrefix: "test"), "10");
    }

    private static JProperty FindNode(WorkflowGenerator generator, string classType)
        => generator.Workflow.Properties().Single(property => property.Value["class_type"]?.Value<string>() == classType);

    [Fact]
    public void TilingWrapsNativeRestoreAndKeepsGlobalColorReference()
    {
        WorkflowGenerator generator = Generator();
        generator.UserInput.Set(T2IParamTypes.ImageFormat, "WEBP");
        TileWorkflow.Run(generator, NativeGraph);
        JProperty split = FindNode(generator, SwarmSeedVRTileSplitNode.ClassType);
        JProperty merge = FindNode(generator, SwarmSeedVRTileMergeNode.ClassType);
        Assert.Equal(new JArray(split.Name, 0), generator.Workflow["2"]["inputs"]["resized_images"]);
        Assert.Equal(new JArray("1", 0), split.Value["inputs"]["image"]);
        Assert.Equal(new JArray("8", 0), merge.Value["inputs"]["images"]);
        Assert.Equal(new JArray(split.Name, 1), merge.Value["inputs"]["layout"]);
        Assert.Equal(new JArray(merge.Name, 0), generator.Workflow["9"]["inputs"]["images"]);
        Assert.Equal(new JArray("1", 0), generator.Workflow["9"]["inputs"]["original_resized_images"]);
        Assert.Equal("wavelet", (string)generator.Workflow["9"]["inputs"]["color_correction_method"]);
        Assert.Equal(new JArray("9", 0), generator.Workflow["10"]["inputs"]["images"]);
        Assert.Equal(42, (long)generator.Workflow["7"]["inputs"]["seed"]);
        Assert.Equal(new JArray("5", 0), generator.Workflow["7"]["inputs"]["model"]);
        Assert.Equal(12, generator.Workflow.Count);
        Assert.True(generator.UserInput.ExtraMeta.ContainsKey(OutputFormatGuard.ContextKey));
        JObject tiled = (JObject)generator.Workflow.DeepClone();
        string next = generator.CreateNode("PreviewImage", new JObject { ["images"] = new JArray(merge.Name, 0) });
        Assert.False(tiled.ContainsKey(next));
        foreach (JProperty property in tiled.Properties())
        {
            Assert.True(JToken.DeepEquals(property.Value, generator.Workflow[property.Name]));
        }
    }

    [Fact]
    public void EarlierSeedVrStagesAreNotRetiled()
    {
        WorkflowGenerator generator = Generator();
        NativeGraph(generator);
        JObject earlier = (JObject)generator.Workflow.DeepClone();
        generator.Workflow.RemoveAll();
        foreach (JProperty property in earlier.Properties())
        {
            JObject node = (JObject)property.Value.DeepClone();
            foreach (JProperty input in ((JObject)node["inputs"]).Properties())
            {
                if (input.Value is JArray connection && connection.Count == 2)
                {
                    connection[0] = "earlier_" + connection[0].Value<string>();
                }
            }
            generator.Workflow["earlier_" + property.Name] = node;
        }
        JObject before = (JObject)generator.Workflow.DeepClone();
        TileWorkflow.Run(generator, NativeGraph);
        foreach (JProperty property in before.Properties())
        {
            Assert.True(JToken.DeepEquals(property.Value, generator.Workflow[property.Name]));
        }
        Assert.Single(generator.Workflow.Properties(), property => property.Value["class_type"]?.Value<string>() == SwarmSeedVRTileSplitNode.ClassType);
    }

    [Fact]
    public void TemporalSplittingDoesNotCollapseSpatialTileList()
    {
        WorkflowGenerator generator = Generator();
        generator.UserInput.Set(ComfyUIBackendExtension.SeedVRSplitLatent, true);
        TileWorkflow.Run(generator, g =>
        {
            NativeGraph(g);
            using WorkflowBridge bridge = WorkflowBridge.Create(g.Workflow);
            VAEEncodeNode encode = bridge.Graph.NodesOfType<VAEEncodeNode>().Single();
            SeedVR2ConditioningNode conditioning = bridge.Graph.NodesOfType<SeedVR2ConditioningNode>().Single();
            KSamplerNode sampler = bridge.Graph.NodesOfType<KSamplerNode>().Single();
            VAEDecodeNode decode = bridge.Graph.NodesOfType<VAEDecodeNode>().Single();
            SeedVR2TemporalChunkNode chunk = bridge.AddNode(new SeedVR2TemporalChunkNode().With(Latent: encode.LATENT), "11");
            conditioning.VaeConditioning.ConnectTo(chunk.Latents);
            sampler.LatentImage.ConnectTo(chunk.Latents);
            SeedVR2TemporalMergeNode merge = bridge.AddNode(new SeedVR2TemporalMergeNode().With(
                Latents: sampler.LATENT, TemporalOverlap: chunk.TemporalOverlap), "12");
            decode.Samples.ConnectTo(merge.Latent);
        });
        Assert.DoesNotContain(generator.Workflow.Properties(), property =>
            property.Value["class_type"]?.Value<string>() is "SeedVR2TemporalChunk" or "SeedVR2TemporalMerge");
        Assert.Equal(new JArray("4", 0), generator.Workflow["6"]["inputs"]["vae_conditioning"]);
        Assert.Equal(new JArray("4", 0), generator.Workflow["7"]["inputs"]["latent_image"]);
        Assert.Equal(new JArray("7", 0), generator.Workflow["8"]["inputs"]["samples"]);
        Assert.True(generator.UserInput.Get(ComfyUIBackendExtension.SeedVRSplitLatent));
    }
}
