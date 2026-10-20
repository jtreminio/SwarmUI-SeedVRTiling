using SeedVRTiling.Generated;
using System.Runtime.CompilerServices;
using ComfyTyped.Core;
using ComfyTyped.Generated;
using ComfyTyped.SwarmUI;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SeedVRTiling;

internal static class TileWorkflow
{
    /// <summary>Captures the graph before core SeedVR restoration at priority 40.</summary>
    internal const double BeforeSeedVRPriority = 39.9;

    /// <summary>Applies tiling after core SeedVR restoration and before final saving.</summary>
    internal const double AfterSeedVRPriority = 40.1;

    /// <summary>Pending state belongs to each generator without retaining abandoned requests.</summary>
    private static readonly ConditionalWeakTable<WorkflowGenerator, HashSet<string>> Pending = new();

    internal static void BeforeSeedVR(WorkflowGenerator generator)
    {
        Pending.Remove(generator);
        if (WorkflowGenerator.RestrictCustomNodes
            || !generator.UserInput.TryGet(SeedVRTilingExtension.Enabled, out bool enabled)
            || !enabled
            || generator.UserInput.Get(T2IParamTypes.JustLoadModel, false)
            || generator.UserInput.Get(T2IParamTypes.BatchSize, 1) > 1
            || generator.CurrentMedia is not { } media
            || (media.DataType != WGNodeData.DT_IMAGE && media.DataType != WGNodeData.DT_LATENT_IMAGE)
            || media.Frames > 1)
        {
            return;
        }
        // Core SeedVR allocates sequential IDs. Other extensions can reserve higher IDs
        // without advancing LastID (for example, an audio save node).
        BridgeSync.SyncLastId(generator);
        Pending.Add(generator, generator.Workflow.Properties().Select(property => property.Name).ToHashSet());
    }

    internal static void AfterSeedVR(WorkflowGenerator generator)
    {
        if (!Pending.TryGetValue(generator, out HashSet<string> existing))
        {
            return;
        }
        Pending.Remove(generator);
        using WorkflowBridge bridge = BridgeSync.For(generator);
        SeedVR2PreprocessNode pre = bridge.Graph.NodesOfType<SeedVR2PreprocessNode>()
            .SingleOrDefault(node => !existing.Contains(node.Id));
        if (pre is null)
        {
            return;
        }
        SeedVR2PostProcessingNode post = bridge.Graph.NodesOfType<SeedVR2PostProcessingNode>()
            .Single(node => !existing.Contains(node.Id));
        if (!generator.Features.Contains(SeedVRTilingExtension.SplitFeature)
            || !generator.Features.Contains(SeedVRTilingExtension.MergeFeature))
        {
            throw new SwarmUserErrorException("SeedVR Image Tiling requires its split and merge nodes on the selected ComfyUI backend. Install the extension's Comfy node package on that backend and restart it.");
        }
        int size = generator.UserInput.Get(SeedVRTilingExtension.TileSize, 512);
        int overlap = generator.UserInput.Get(SeedVRTilingExtension.TileOverlap, 32);
        if (overlap >= size)
        {
            throw new SwarmUserErrorException("SeedVR tile overlap must be smaller than tile size.");
        }
        foreach (SeedVR2TemporalMergeNode temporal in bridge.Graph.NodesOfType<SeedVR2TemporalMergeNode>()
            .Where(node => !existing.Contains(node.Id)).ToArray())
        {
            bridge.Graph.RetargetConnections(temporal.Latent, temporal.Latents.Connection);
            bridge.RemoveNode(temporal);
        }
        foreach (SeedVR2TemporalChunkNode temporal in bridge.Graph.NodesOfType<SeedVR2TemporalChunkNode>()
            .Where(node => !existing.Contains(node.Id)).ToArray())
        {
            bridge.Graph.RetargetConnections(temporal.Latents, temporal.Latent.Connection);
            bridge.RemoveNode(temporal);
        }
        SwarmSeedVRTileSplitNode split = bridge.AddNode(new SwarmSeedVRTileSplitNode());
        split.Image.ConnectSameAs(pre.ResizedImages);
        split.TileSize.Set(size);
        split.TileOverlap.Set(overlap);
        pre.ResizedImages.ConnectTo(split.Tiles);

        SwarmSeedVRTileMergeNode merge = bridge.AddNode(new SwarmSeedVRTileMergeNode());
        merge.Images.ConnectSameAs(post.ImagesInput);
        merge.Layout.ConnectTo(split.Layout);
        merge.AntiAliasing.Set(generator.UserInput.Get(SeedVRTilingExtension.AntiAliasing, 0));
        merge.Blending.Set(generator.UserInput.Get(SeedVRTilingExtension.Blending, "multiband"));
        post.ImagesInput.ConnectTo(merge.IMAGE);
        OutputFormatGuard.Prepare(generator.UserInput);
    }
}
