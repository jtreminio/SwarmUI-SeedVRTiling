using SixLabors.ImageSharp.PixelFormats;
using System.Runtime.CompilerServices;
using SwarmUI.Accounts;
using SwarmUI.Media;
using SwarmUI.Text2Image;
using SwarmUI.Utils;
using Xunit;

namespace SeedVRTiling.Tests;

public class OutputFormatTests
{
    static OutputFormatTests() => WorkflowTests.EnsureInitialized();

    private static T2IParamInput Input(string fallback = "PNG", string format = "WEBP")
    {
        T2IParamInput input = new(null);
        input.Set(T2IParamTypes.ImageFormat, format);
        input.Set(SeedVRTilingExtension.MegaDimensionsFallback, fallback);
        return input;
    }

    private static ImageFile Image(int width, int height)
    {
        using SixLabors.ImageSharp.Image<Rgba32> image = new(width, height);
        return new SwarmUI.Utils.Image(ImageFile.ISImgToPngBytes(image), MediaType.ImagePng);
    }

    [Theory]
    [InlineData("WEBP", "Off", 16383, 16383, "WEBP")]
    [InlineData("WEBP", "JPG", 13312, 19456, "JPG")]
    [InlineData("WEBP", "Off", 16384, 2, "PNG")]
    [InlineData("WEBP", "invalid", 16384, 2, "PNG")]
    [InlineData("WEBP", "JPG", 65536, 2, "PNG")]
    [InlineData("JPG", "JPG", 65535, 2, "JPG")]
    [InlineData("JPG", "JPG", 65536, 2, "PNG")]
    [InlineData("JPG_90", "PNG", 2, 65536, "PNG")]
    [InlineData("JPEG", "PNG", 2, 65536, "PNG")]
    [InlineData("WEBP_LOSSLESS", "PNG", 13312, 19456, "PNG")]
    [InlineData("WEBP_90", "PNG", 64, 16384, "PNG")]
    [InlineData("WEBP_90", "PNG", 100, 100, "WEBP_90")]
    [InlineData("JPG", "Off", 13312, 19456, "JPG")]
    [InlineData("PNG", "Off", 13312, 19456, "PNG")]
    public void ChoosesFallbackOnlyWhenFormatExceedsLimits(string format, string fallback, int width, int height, string expected)
        => Assert.Equal(expected, OutputFormatGuard.SelectFormat(format, fallback, width, height));

    [Theory]
    [InlineData("JPG", "jpg")]
    [InlineData("PNG", "png")]
    public void ActualOutputUpdatesEncodingAndFilenameRequestsAndKeepsMetadata(string fallback, string extension)
    {
        T2IParamInput request = Input(fallback);
        OutputFormatGuard.Prepare(request);
        T2IParamInput metadataInput = request.Clone();
        ImageFile source = Image(2, 16384);
        OutputFormatGuard.HandleOutput(new(source, metadataInput, () => { }));
        Assert.Equal(fallback, metadataInput.Get(T2IParamTypes.ImageFormat));
        Assert.Equal(fallback, request.Get(T2IParamTypes.ImageFormat));
        Assert.False(metadataInput.ExtraMeta.ContainsKey(OutputFormatGuard.ContextKey));
        const string metadata = "{\"sui_image_params\":{\"prompt\":\"fallback test\"}}";
        ImageFile converted = source.ConvertTo(metadataInput.Get(T2IParamTypes.ImageFormat), metadata);
        Assert.Equal(extension, converted.Type.Extension);
        Assert.Equal(extension, ImageFile.ImageFormatToExtension(request.Get(T2IParamTypes.ImageFormat)));
        Assert.Equal((2, 16384), converted.GetResolution());
        Assert.Equal(metadata, converted.GetMetadata());
        // A following small intermediate should still use the selected WebP
        // variant, even though the preceding mega-image used its fallback.
        T2IParamInput small = request.Clone();
        OutputFormatGuard.HandleOutput(new(Image(2, 2), small, () => { }));
        Assert.Equal("WEBP", small.Get(T2IParamTypes.ImageFormat));
        Assert.Equal("WEBP", request.Get(T2IParamTypes.ImageFormat));
    }

    [Theory]
    [InlineData("WEBP", 16384)]
    [InlineData("JPG", 65536)]
    public void ActualOversizedOutputFallsBackWithoutKnownWorkflowDimensions(string format, int width)
    {
        T2IParamInput input = Input(format: format);
        OutputFormatGuard.Prepare(input);
        T2IParamInput output = input.Clone();
        OutputFormatGuard.HandleOutput(new(Image(width, 2), output, () => { }));
        Assert.Equal("PNG", output.Get(T2IParamTypes.ImageFormat));
        Assert.Equal("PNG", input.Get(T2IParamTypes.ImageFormat));
    }

    [Fact]
    public void UserDefaultIsUsedUnlessRequestOverridesIt()
    {
        User user = (User)RuntimeHelpers.GetUninitializedObject(typeof(User));
        user.Settings = new();
        user.Settings.FileFormat.ImageFormat = "WEBP_LOSSLESS";
        T2IParamInput input = new(new Session { User = user });
        OutputFormatGuard.Prepare(input);
        T2IParamInput output = input.Clone();
        OutputFormatGuard.HandleOutput(new(Image(16384, 2), output, () => { }));
        Assert.Equal("PNG", output.Get(T2IParamTypes.ImageFormat));
        input.Set(T2IParamTypes.ImageFormat, "PNG");
        OutputFormatGuard.Prepare(input);
        output = input.Clone();
        OutputFormatGuard.HandleOutput(new(Image(2, 2), output, () => { }));
        Assert.Equal("PNG", output.Get(T2IParamTypes.ImageFormat));
        Assert.Equal("WEBP_LOSSLESS", user.Settings.FileFormat.ImageFormat);
    }
}
