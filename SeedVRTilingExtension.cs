using System.IO;
using ComfyTyped.Core;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;

namespace SeedVRTiling;

/// <summary>Adds spatial image tiling to SwarmUI's native SeedVR restoration stage.</summary>
public sealed class SeedVRTilingExtension : Extension
{
    public static T2IRegisteredParam<bool> Enabled;
    public static T2IRegisteredParam<int> TileSize, TileOverlap;
    public static T2IRegisteredParam<double> AntiAliasing;
    public static T2IRegisteredParam<string> Blending, MegaDimensionsFallback;
    private static bool hookInstalled;

    public override void OnPreInit()
    {
        ScriptFiles.Add("Assets/seedvr-tiling.js");
        ComfyUIBackendExtension.NodeToFeatureMap[TileSplitNode.ClassType] = TileSplitNode.ClassType;
        ComfyUIBackendExtension.NodeToFeatureMap[TileMergeNode.ClassType] = TileMergeNode.ClassType;
        ComfyUIBackendExtension.NodeToFeatureMap[TileProgressModelNode.ClassType] = TileProgressModelNode.ClassType;
    }

    public override void OnInit()
    {
        ComfyTyped.Generated.NodeRegistrations.EnsureRegistered();
        NodeRegistry.RegisterAssembly(typeof(TileSplitNode).Assembly);
        ComfyUISelfStartBackend.CustomNodePaths.Add(Path.GetFullPath(Path.Join(FilePath, "comfy_node")));
        RegisterParameters();
        T2IEngine.PostGenerateEvent += OutputFormatGuard.HandleOutput;
    }

    internal static void RegisterParameters()
    {
        T2IParamGroup group = ComfyUIBackendExtension.GroupSeedVR
            ?? throw new InvalidOperationException("SeedVR Tiling requires SwarmUI with native SeedVR support.");
        Enabled = T2IParamTypes.Register<bool>(new("SeedVR Image Tiling",
            "If enabled, restores the image through SeedVR in smaller, overlapping tiles.\n"
            + "Tiling reduces VRAM use, but takes longer and may affect quality. For still images only.",
            "false", IgnoreIf: "false", Group: group, FeatureFlag: "comfyui", OrderPriority: 4, DoNotPreview: true));
        string depends = Enabled.Type.ID;
        TileSize = T2IParamTypes.Register<int>(new("SeedVR Image Tile Size",
            "Restores the image through SeedVR using tiles of this size, measured in pixels after upscaling.\n"
            + "Smaller tiles use less VRAM, but take longer and may affect quality. Larger tiles give SeedVR more image context, but need more VRAM.",
            "512", Min: 64, Max: 8192, ViewMax: 2048, Step: 8, Group: group, FeatureFlag: "comfyui",
            DependNonDefault: depends, OrderPriority: 5, DoNotPreview: true));
        TileOverlap = T2IParamTypes.Register<int>(new("SeedVR Image Tile Overlap",
            "Controls how much neighboring tiles overlap, in pixels.\n"
            + "Higher overlap can improve quality at tile edges, but uses more VRAM and takes longer. Must be smaller than Tile Size.",
            "32", Min: 0, Max: 8192, ViewMax: 256, Step: 8, Group: group, FeatureFlag: "comfyui",
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
            "WebP allows up to 16,383 pixels per side. Fallback applies only if the width or height exceeds that limit, regardless of total pixel count.\n"
            + "Off: stop with an error asking you to reduce the image size.\n"
            + "JPG: keep the full size, with usually smaller files and some compression loss. Uses your image quality setting; transparency is lost.\n"
            + "PNG: keep the full size and transparency, with lossless compression. Files can be much larger.",
            "Off", GetValues: _ => ["Off", "JPG", "PNG"], Group: group, FeatureFlag: "comfyui",
            DependNonDefault: depends, OrderPriority: 9, DoNotPreview: true));
    }

    public override void OnShutdown()
    {
        T2IEngine.PostGenerateEvent -= OutputFormatGuard.HandleOutput;
    }

    public override void OnPreLaunch()
    {
        if (hookInstalled)
        {
            return;
        }
        // Initialize WorkflowGenerator before reading SeedVRStep. Its static constructor
        // registers core steps. Wrap both references: SeedVR Only invokes the delegate
        // directly and exits before ordinary extension steps can run.
        List<WorkflowGenerator.WorkflowGenStep> steps = WorkflowGenerator.Steps;
        Action<WorkflowGenerator> original = WorkflowGeneratorSteps.SeedVRStep
            ?? throw new InvalidOperationException("Native SwarmUI SeedVR step is unavailable.");
        int index = steps.FindIndex(step => step.Action == original);
        if (index < 0)
        {
            throw new InvalidOperationException("Cannot locate the native SeedVR workflow step.");
        }
        Action<WorkflowGenerator> wrapped = g => TileWorkflow.Run(g, original);
        WorkflowGeneratorSteps.SeedVRStep = wrapped;
        steps[index] = new(wrapped, steps[index].Priority);
        hookInstalled = true;
    }
}
