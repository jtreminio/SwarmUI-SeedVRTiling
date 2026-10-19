using Newtonsoft.Json;
using SixLabors.ImageSharp;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SeedVRTiling;

/// <summary>Selects an encodable image format before metadata conversion and saving.</summary>
internal static class OutputFormatGuard
{
    internal const int WebpMaxDimension = 16383;
    internal const string ContextKey = "seedvr_output_format_context";

    // Preserve the request through per-output clones without serializing its reference.
    [JsonObject(MemberSerialization.OptIn)]
    private sealed class RequestContext(T2IParamInput request, string format, string fallback)
    {
        internal readonly T2IParamInput Request = request;
        internal readonly string Format = format;
        internal readonly string Fallback = fallback;
    }

    internal const int JpgMaxDimension = 65535;
    internal const string DefaultFallback = "PNG";

    internal static string SelectFormat(string format, string fallback, int width, int height)
    {
        bool tooLargeForJpg = width > JpgMaxDimension || height > JpgMaxDimension;
        if (format.StartsWith("JPG", StringComparison.OrdinalIgnoreCase)
            || format.StartsWith("JPEG", StringComparison.OrdinalIgnoreCase))
        {
            return tooLargeForJpg ? DefaultFallback : format;
        }
        if (!format.StartsWith("WEBP", StringComparison.OrdinalIgnoreCase)
            || (width <= WebpMaxDimension && height <= WebpMaxDimension))
        {
            return format;
        }
        return fallback == "JPG" && !tooLargeForJpg ? "JPG" : DefaultFallback;
    }

    internal static void Prepare(T2IParamInput input)
    {
        string format = input.Get(T2IParamTypes.ImageFormat, input.SourceSession?.User.Settings.FileFormat.ImageFormat ?? "PNG");
        string fallback = input.Get(SeedVRTilingExtension.MegaDimensionsFallback, DefaultFallback);
        input.ExtraMeta[ContextKey] = new RequestContext(input, format, fallback);
    }

    internal static void HandleOutput(T2IEngine.PostGenerationEventParams output)
    {
        if (!output.UserInput.ExtraMeta.Remove(ContextKey, out object raw)
            || raw is not RequestContext context || output.File?.Type.MetaType != MediaMetaType.Image)
        {
            return;
        }
        // Read dimensions without decoding the full image.
        ImageInfo info = SixLabors.ImageSharp.Image.Identify(output.File.RawData);
        string format = SelectFormat(context.Format, context.Fallback, info.Width, info.Height);
        if (output.File is ImageFile image)
        {
            LargeWebpEncoding.Prepare(image, format, info.Width, info.Height);
        }
        // ApplyMetadata receives the per-output clone, but SaveImage receives the
        // original request. Set both so encoded bytes and the filename agree.
        output.UserInput.Set(T2IParamTypes.ImageFormat, format);
        context.Request.Set(T2IParamTypes.ImageFormat, format);
        if (format != context.Format)
        {
            output.UserInput.ExtraMeta["seedvr_requested_image_format"] = context.Format;
            Logs.Info($"[SeedVR Tiling] {info.Width}x{info.Height} exceeds {context.Format} dimensions; saving as {format} per fallback setting.");
        }
    }
}
