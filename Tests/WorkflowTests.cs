using SeedVRTiling.Generated;
using ComfyTyped.Core;
using ComfyTyped.Generated;
using Newtonsoft.Json.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
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
        T2IParamTypes.JustLoadModel = T2IParamTypes.Register<bool>(new("Just Load Model", "", "false"));
        T2IParamTypes.BatchSize = T2IParamTypes.Register<int>(new("Batch Size", "", "1", Min: 1, Max: 100, Step: 1));
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
        generator.CurrentMedia = new WGNodeData(new JArray("source", 0), generator, WGNodeData.DT_IMAGE, null);
        generator.Features.UnionWith([SeedVRTilingExtension.SplitFeature, SeedVRTilingExtension.MergeFeature]);
        generator.UserInput.Set(SeedVRTilingExtension.Enabled, true);
        generator.UserInput.Set(ComfyUIBackendExtension.SeedVRModel, new T2IModel(null, "", "seedvr.safetensors", "seedvr"));
        return generator;
    }

    private static void RunStage(WorkflowGenerator generator, Action<WorkflowGenerator> nativeStep)
    {
        TileWorkflow.BeforeSeedVR(generator);
        nativeStep(generator);
        TileWorkflow.AfterSeedVR(generator);
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
        RunStage(generator, NativeGraph);
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
        RunStage(generator, NativeGraph);
        foreach (JProperty property in before.Properties())
        {
            Assert.True(JToken.DeepEquals(property.Value, generator.Workflow[property.Name]));
        }
        Assert.Single(generator.Workflow.Properties(), property => property.Value["class_type"]?.Value<string>() == SwarmSeedVRTileSplitNode.ClassType);
    }

    [Theory]
    [InlineData(65162)]
    [InlineData(70000)]
    public void NativeRestorePreservesReservedAudioOutputIds(int nextId)
    {
        WorkflowGenerator generator = Generator();
        generator.CreateNode("SaveAudio", new JObject
        {
            ["audio"] = new JArray("65160", 0),
            ["filename_prefix"] = "audio"
        }, "65170");
        generator.LastID = nextId;
        JObject audioOutput = (JObject)generator.Workflow["65170"].DeepClone();

        // Core allocates sequential IDs without checking explicitly reserved IDs.
        // With LastID 65162, the ninth node (postprocessing) overwrites 65170.
        WorkflowGenerator template = Generator();
        NativeGraph(template);
        Dictionary<string, string> ids = [];
        RunStage(generator, g =>
        {
            foreach (JProperty property in template.Workflow.Properties())
            {
                JObject inputs = (JObject)property.Value["inputs"].DeepClone();
                foreach (JProperty input in inputs.Properties())
                {
                    if (input.Value is JArray connection)
                    {
                        connection[0] = ids[connection[0].Value<string>()];
                    }
                }
                ids[property.Name] = g.CreateNode(property.Value["class_type"].Value<string>(), inputs);
            }
        });

        Assert.True(JToken.DeepEquals(audioOutput, generator.Workflow["65170"]));
        Assert.True(int.Parse(ids["1"]) >= Math.Max(nextId, 65171));
        JProperty split = FindNode(generator, SwarmSeedVRTileSplitNode.ClassType);
        JProperty merge = FindNode(generator, SwarmSeedVRTileMergeNode.ClassType);
        Assert.Equal(new JArray(split.Name, 0), generator.Workflow[ids["2"]]["inputs"]["resized_images"]);
        Assert.Equal(new JArray(ids["8"], 0), merge.Value["inputs"]["images"]);
        Assert.Equal(new JArray(merge.Name, 0), generator.Workflow[ids["9"]]["inputs"]["images"]);
        Assert.Equal(new JArray(ids["9"], 0), generator.Workflow[ids["10"]]["inputs"]["images"]);
        Assert.Equal(13, generator.Workflow.Count);
    }

    [Fact]
    public void TemporalSplittingDoesNotCollapseSpatialTileList()
    {
        WorkflowGenerator generator = Generator();
        generator.UserInput.Set(ComfyUIBackendExtension.SeedVRSplitLatent, true);
        RunStage(generator, g =>
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

    [Theory]
    [InlineData("IMAGE")]
    [InlineData("LATENT_IMAGE")]
    public void HooksRunAroundNativeRestoreBeforeFinalSave(string dataType)
    {
        WorkflowGenerator generator = Generator();
        generator.CurrentMedia.DataType = dataType;
        List<WorkflowGenerator.WorkflowGenStep> steps =
        [
            new(TileWorkflow.AfterSeedVR, TileWorkflow.AfterSeedVRPriority),
            new(g =>
            {
                Assert.NotNull(FindNode(g, SwarmSeedVRTileMergeNode.ClassType));
            }, 100),
            new(NativeGraph, 40),
            new(TileWorkflow.BeforeSeedVR, TileWorkflow.BeforeSeedVRPriority)
        ];
        foreach (WorkflowGenerator.WorkflowGenStep step in steps.OrderBy(step => step.Priority))
        {
            step.Action(generator);
        }
        Assert.Equal(12, generator.Workflow.Count);
    }

    [Fact]
    public void MissingSeedVrModelLeavesNativeGraphUnchanged()
    {
        WorkflowGenerator generator = Generator();
        generator.Features.Clear();
        generator.UserInput.Remove(ComfyUIBackendExtension.SeedVRModel);
        NativeGraph(generator);
        JObject existing = (JObject)generator.Workflow.DeepClone();
        RunStage(generator, WorkflowGenerator.Steps.Single(step => step.Priority == 40).Action);
        Assert.True(JToken.DeepEquals(existing, generator.Workflow));
        Assert.False(generator.UserInput.ExtraMeta.ContainsKey(OutputFormatGuard.ContextKey));
    }

    [Theory]
    [InlineData("disabled")]
    [InlineData("absent")]
    [InlineData("load-model")]
    [InlineData("batch")]
    [InlineData("frames")]
    [InlineData("VIDEO")]
    [InlineData("LATENT_VIDEO")]
    [InlineData("AUDIO")]
    [InlineData("LATENT_AUDIO")]
    [InlineData("no-media")]
    public void IneligibleRequestsKeepNativeGraphUnchanged(string scenario)
    {
        WorkflowGenerator generator = Generator();
        switch (scenario)
        {
            case "disabled":
                generator.UserInput.Set(SeedVRTilingExtension.Enabled, false);
                break;
            case "absent":
                generator.UserInput.Remove(SeedVRTilingExtension.Enabled);
                break;
            case "load-model":
                generator.UserInput.Set(T2IParamTypes.JustLoadModel, true);
                break;
            case "batch":
                generator.UserInput.Set(T2IParamTypes.BatchSize, 2);
                break;
            case "frames":
                generator.CurrentMedia.Frames = 2;
                break;
            case "no-media":
                generator.CurrentMedia = null;
                break;
            default:
                generator.CurrentMedia.DataType = scenario;
                break;
        }
        // Ineligible requests must not require extension nodes on the backend.
        generator.Features.Clear();
        JObject original = null;
        RunStage(generator, g =>
        {
            NativeGraph(g);
            original = (JObject)g.Workflow.DeepClone();
        });
        Assert.True(JToken.DeepEquals(original, generator.Workflow));
        Assert.False(generator.UserInput.ExtraMeta.ContainsKey(OutputFormatGuard.ContextKey));
    }

    [Fact]
    public void RestrictedCustomNodesLeaveNativeGraphUnchanged()
    {
        bool previous = WorkflowGenerator.RestrictCustomNodes;
        try
        {
            WorkflowGenerator.RestrictCustomNodes = true;
            WorkflowGenerator generator = Generator();
            generator.Features.Clear();
            RunStage(generator, NativeGraph);
            Assert.Equal(10, generator.Workflow.Count);
            Assert.False(generator.UserInput.ExtraMeta.ContainsKey(OutputFormatGuard.ContextKey));
        }
        finally
        {
            WorkflowGenerator.RestrictCustomNodes = previous;
        }
    }

    [Theory]
    [InlineData(SeedVRTilingExtension.SplitFeature)]
    [InlineData(SeedVRTilingExtension.MergeFeature)]
    public void MissingBackendNodeFailsBeforeTilingChanges(string feature)
    {
        WorkflowGenerator generator = Generator();
        generator.Features.Remove(feature);
        JObject original = null;
        SwarmUserErrorException error = Assert.Throws<SwarmUserErrorException>(() =>
            RunStage(generator, g =>
            {
                NativeGraph(g);
                original = (JObject)g.Workflow.DeepClone();
            }));
        Assert.Contains("selected ComfyUI backend", error.Message);
        Assert.True(JToken.DeepEquals(original, generator.Workflow));
    }

    [Fact]
    public void InvalidOverlapFailsBeforeTilingChanges()
    {
        WorkflowGenerator generator = Generator();
        generator.UserInput.Set(SeedVRTilingExtension.TileOverlap, 512);
        JObject original = null;
        Assert.Throws<SwarmUserErrorException>(() => RunStage(generator, g =>
        {
            NativeGraph(g);
            original = (JObject)g.Workflow.DeepClone();
        }));
        Assert.True(JToken.DeepEquals(original, generator.Workflow));
    }

    [Fact]
    public void InterleavedGeneratorsKeepTheirOwnTileSettings()
    {
        WorkflowGenerator first = Generator();
        WorkflowGenerator second = Generator();
        first.UserInput.Set(SeedVRTilingExtension.TileSize, 1024);
        second.UserInput.Set(SeedVRTilingExtension.TileSize, 256);
        TileWorkflow.BeforeSeedVR(first);
        TileWorkflow.BeforeSeedVR(second);
        NativeGraph(second);
        TileWorkflow.AfterSeedVR(second);
        NativeGraph(first);
        TileWorkflow.AfterSeedVR(first);
        Assert.Equal(1024, (int)FindNode(first, SwarmSeedVRTileSplitNode.ClassType).Value["inputs"]["tile_size"]);
        Assert.Equal(256, (int)FindNode(second, SwarmSeedVRTileSplitNode.ClassType).Value["inputs"]["tile_size"]);
        JObject completed = (JObject)first.Workflow.DeepClone();
        TileWorkflow.AfterSeedVR(first);
        Assert.True(JToken.DeepEquals(completed, first.Workflow));
    }

    [Fact]
    public void NoNewNativeRestorationLeavesEarlierStagesUntouchedAndClearsPendingState()
    {
        WorkflowGenerator generator = Generator();
        generator.Features.Clear();
        generator.UserInput.Set(SeedVRTilingExtension.TileOverlap, 512);
        NativeGraph(generator);
        JObject before = (JObject)generator.Workflow.DeepClone();
        TileWorkflow.BeforeSeedVR(generator);
        TileWorkflow.AfterSeedVR(generator);
        Assert.True(JToken.DeepEquals(before, generator.Workflow));
        Assert.False(generator.UserInput.ExtraMeta.ContainsKey(OutputFormatGuard.ContextKey));
        generator.Workflow.RemoveAll();
        NativeGraph(generator);
        TileWorkflow.AfterSeedVR(generator);
        Assert.Equal(10, generator.Workflow.Count);
    }

}
