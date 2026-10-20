using SeedVRTiling.Generated;
using System.IO;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;

namespace SeedVRTiling;

/// <summary>Adds opt-in spatial tiling to Swarm's native SeedVR restoration stage.</summary>
public sealed class SeedVRTilingExtension : Extension
{
    /// <summary>Backend capability discovered from the tile split node.</summary>
    internal const string SplitFeature = "seedvr_tiling_split";

    /// <summary>Backend capability discovered from the tile merge node.</summary>
    internal const string MergeFeature = "seedvr_tiling_merge";

    /// <summary>Enables tiling while preserving the existing preset and API parameter ID.</summary>
    public static T2IRegisteredParam<bool> Enabled;

    /// <summary>Tile side length in pixels after upscaling.</summary>
    public static T2IRegisteredParam<int> TileSize;

    /// <summary>Overlap between neighboring tiles in pixels.</summary>
    public static T2IRegisteredParam<int> TileOverlap;

    /// <summary>Optional smoothing strength after joining tiles.</summary>
    public static T2IRegisteredParam<double> AntiAliasing;

    /// <summary>Method used to blend overlapping tiles.</summary>
    public static T2IRegisteredParam<string> Blending;

    /// <summary>Output format used when the requested format cannot encode the image.</summary>
    public static T2IRegisteredParam<string> MegaDimensionsFallback;

    public override void OnInit()
    {
        ScriptFiles.Add("Assets/seedvr-tiling.js");
        ComfyTyped.Generated.NodeRegistrations.EnsureRegistered();
        TileNodeRegistrations.EnsureRegistered();
        ComfyUISelfStartBackend.CustomNodePaths.Add(Path.GetFullPath(Path.Join(FilePath, "comfy_node")));
        ComfyUIBackendExtension.NodeToFeatureMap[SwarmSeedVRTileSplitNode.ClassType] = SplitFeature;
        ComfyUIBackendExtension.NodeToFeatureMap[SwarmSeedVRTileMergeNode.ClassType] = MergeFeature;
        RegisterParameters();
        WorkflowGenerator.AddStep(TileWorkflow.BeforeSeedVR, TileWorkflow.BeforeSeedVRPriority);
        WorkflowGenerator.AddStep(TileWorkflow.AfterSeedVR, TileWorkflow.AfterSeedVRPriority);
        T2IEngine.PostGenerateEvent += OutputFormatGuard.HandleOutput;
    }

    internal static void RegisterParameters()
    {
        T2IParamGroup group = ComfyUIBackendExtension.GroupSeedVR;
        Enabled = T2IParamTypes.Register<bool>(new("SeedVR Image Tiling",
            "If enabled, restores the image through SeedVR in smaller, overlapping tiles.\n"
            + "Tiling reduces VRAM use, but takes longer and may affect quality. For still images only.",
            "false", IgnoreIf: "false", Group: group, FeatureFlag: "comfyui", OrderPriority: 4, DoNotPreview: true));
        string depends = Enabled.Type.ID;
        TileSize = T2IParamTypes.Register<int>(new("SeedVR Image Tile Size",
            "Restores the image through SeedVR using tiles of this size, measured in pixels after upscaling.\n"
            + "Smaller tiles use less VRAM, but take longer and may affect quality. Larger tiles give SeedVR more image context, but need more VRAM.",
            "512", Min: 64, Max: int.MaxValue, ViewMax: 2048, Step: 8, Group: group, FeatureFlag: "comfyui",
            DependNonDefault: depends, OrderPriority: 5, DoNotPreview: true));
        TileOverlap = T2IParamTypes.Register<int>(new("SeedVR Image Tile Overlap",
            "Controls how much neighboring tiles overlap, in pixels.\n"
            + "Higher overlap can improve quality at tile edges, but uses more VRAM and takes longer. Must be smaller than Tile Size.",
            "32", Min: 0, Max: int.MaxValue, ViewMax: 256, Step: 8, Group: group, FeatureFlag: "comfyui",
            DependNonDefault: depends, OrderPriority: 6, DoNotPreview: true));
        AntiAliasing = T2IParamTypes.Register<double>(new("SeedVR Image Anti-Aliasing Strength",
            "Adds smoothing after the tiles are joined.\n"
            + "0 = off, preserving the most detail. Try a low value if edges look jagged. Higher values can soften texture and fine detail.\n"
            + "Adds processing time; it does not improve every image.", "0", Min: 0, Max: 1, Step: 0.05,
            ViewType: ParamViewType.SLIDER, Group: group, FeatureFlag: "comfyui", IsAdvanced: true,
            DependNonDefault: depends, OrderPriority: 8, DoNotPreview: true));
        Blending = T2IParamTypes.Register<string>(new("SeedVR Image Blending Method",
            "Controls how overlapping tiles are joined.\n"
            + "Multiband: use when you have plenty of free VRAM. Uses more memory and GPU work to smooth tile seams and color transitions.\n"
            + "Simple: use for lower resource use. Faster, with less memory and GPU work, but differences between tiles may stay visible.",
            "multiband", GetValues: _ => ["multiband///Multiband (seam smoothing, more VRAM)",
                "simple///Simple (fast, less VRAM)"],
            Group: group, FeatureFlag: "comfyui", IsAdvanced: true,
            DependNonDefault: depends, OrderPriority: 7, DoNotPreview: true));
        MegaDimensionsFallback = T2IParamTypes.Register<string>(new("Fallback on mega-dimensions",
            "Choose the output format when an image exceeds WebP's 16,383-pixel side limit.\n"
            + "PNG preserves detail and transparency. JPG produces smaller files with compression loss and no transparency.\n"
            + "Images exceeding JPG's 65,535-pixel side limit always use PNG.",
            OutputFormatGuard.DefaultFallback, GetValues: _ => ["PNG", "JPG"],
            Clean: (_, value) => string.Equals(value, "JPG", StringComparison.OrdinalIgnoreCase) ? "JPG" : OutputFormatGuard.DefaultFallback,
            Group: group, FeatureFlag: "comfyui",
            DependNonDefault: depends, OrderPriority: 9, DoNotPreview: true));
    }

    public override void OnShutdown()
    {
        T2IEngine.PostGenerateEvent -= OutputFormatGuard.HandleOutput;
    }

}
