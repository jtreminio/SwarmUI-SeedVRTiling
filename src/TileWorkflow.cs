using SeedVRTiling.Generated;
using ComfyTyped.Core;
using ComfyTyped.Generated;
using ComfyTyped.SwarmUI;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Utils;

namespace SeedVRTiling;

internal static class TileWorkflow
{
    internal static void Run(WorkflowGenerator generator, Action<WorkflowGenerator> nativeStep)
    {
        bool enabled = !WorkflowGenerator.RestrictCustomNodes
            && generator.UserInput.Get(SeedVRTilingExtension.Enabled, false)
            && generator.UserInput.Get(ComfyUIBackendExtension.SeedVRModel) is not null;
        if (!enabled || generator.CurrentMedia is { } media
            && (media.DataType != WGNodeData.DT_IMAGE || media.Frames > 1))
        {
            nativeStep(generator);
            return;
        }
        int size = generator.UserInput.Get(SeedVRTilingExtension.TileSize, 512);
        int overlap = generator.UserInput.Get(SeedVRTilingExtension.TileOverlap, 32);
        if (overlap >= size)
        {
            throw new SwarmReadableErrorException("SeedVR tile overlap must be smaller than tile size.");
        }
        HashSet<string> existing = generator.Workflow.Properties().Select(property => property.Name).ToHashSet();
        nativeStep(generator);
        OutputFormatGuard.Prepare(generator.UserInput);
        using WorkflowBridge bridge = BridgeSync.For(generator);
        SeedVR2PreprocessNode pre = bridge.Graph.NodesOfType<SeedVR2PreprocessNode>()
            .Single(node => !existing.Contains(node.Id));
        SeedVR2PostProcessingNode post = bridge.Graph.NodesOfType<SeedVR2PostProcessingNode>()
            .Single(node => !existing.Contains(node.Id));
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
    }
}
