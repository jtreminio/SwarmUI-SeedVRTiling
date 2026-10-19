using System.Buffers;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Webp;
using SixLabors.ImageSharp.Memory;
using SixLabors.ImageSharp.PixelFormats;
using SwarmUI.Media;
using Xunit;

namespace SeedVRTiling.Tests;

public class LargeWebpEncodingTests
{
    [Fact]
    public void AllocatorPreservesCleanBuffersAndDisposal()
    {
        MemoryAllocator limited = MemoryAllocator.Create(new() { AllocationLimitMegabytes = 1 });
        LargeWebpAllocator allocator = new(limited, 1024 * 1024);
        IMemoryOwner<ulong> buffer = allocator.Allocate<ulong>(262144, AllocationOptions.Clean);
        Assert.Equal(262144, buffer.Memory.Length);
        Assert.Equal(0UL, buffer.Memory.Span[0]);
        Assert.Equal(0UL, buffer.Memory.Span[^1]);
        buffer.Dispose();
        buffer.Dispose();
        Assert.Throws<ObjectDisposedException>(() => buffer.Memory);
    }

    [Fact]
    public void RealWebpAlphaEncodingSucceedsPastDelegateBufferLimit()
    {
        // Reproduce the reported Vp8LBackwardRefs failure with a 1 MiB limit
        // and a small image, rather than allocating gigabytes in unit tests.
        MemoryAllocator defaultAllocator = Configuration.Default.MemoryAllocator;
        MemoryAllocator limited = MemoryAllocator.Create(new() { AllocationLimitMegabytes = 1 });
        Configuration configuration = Configuration.Default.Clone();
        configuration.MemoryAllocator = limited;
        using Image<Rgba32> image = new(configuration, 512, 512, new Rgba32(100, 140, 180, 254));
        using MemoryStream failing = new();
        InvalidMemoryOperationException error = Assert.Throws<InvalidMemoryOperationException>(() =>
            image.SaveAsWebp(failing, new WebpEncoder { FileFormat = WebpFileFormatType.Lossy }));
        Assert.Contains("Vp8LBackwardRefs", error.StackTrace);

        ImageFile source = new SwarmUI.Utils.Image(ImageFile.ISImgToPngBytes(image), MediaType.ImagePng);
        LargeWebpEncoding.Configure(source, new LargeWebpAllocator(limited, 1024 * 1024));
        try
        {
            const string metadata = "{\"sui_image_params\":{\"prompt\":\"large WebP test\"}}";
            ImageFile converted = source.ConvertTo("WEBP", metadata, quality: 90, stealthMetadata: "alpha");
            try
            {
                Assert.Equal("webp", converted.Type.Extension);
                Assert.Equal((512, 512), converted.GetResolution());
                Assert.Equal(metadata, converted.GetMetadata());
                using Image<Rgba32> decoded = Image.Load<Rgba32>(converted.RawData);
                Assert.Equal(254, decoded[511, 0].A);
                Assert.IsType<LargeWebpAllocator>(source.ToIS.Configuration.MemoryAllocator);
                Assert.Same(defaultAllocator, Configuration.Default.MemoryAllocator);
            }
            finally
            {
                converted._CacheISImg?.Dispose();
            }
        }
        finally
        {
            source._CacheISImg?.Dispose();
        }
    }

    [Fact]
    public void ExistingDecodedCacheUsesPrivateConfigurationAndKeepsPixels()
    {
        using Image<Rgb24> image = new(16, 16, new Rgb24(10, 20, 30));
        ImageFile source = new SwarmUI.Utils.Image(ImageFile.ISImgToPngBytes(image), MediaType.ImagePng);
        SixLabors.ImageSharp.Image original = source.ToIS;
        Configuration originalConfiguration = original.Configuration;
        MemoryAllocator originalAllocator = originalConfiguration.MemoryAllocator;
        LargeWebpEncoding.Configure(source);
        try
        {
            Assert.NotSame(original, source.ToIS);
            Assert.NotSame(originalConfiguration, source.ToIS.Configuration);
            Assert.Same(originalAllocator, originalConfiguration.MemoryAllocator);
            using Image<Rgb24> pixels = source.ToIS.CloneAs<Rgb24>();
            Assert.Equal(new Rgb24(10, 20, 30), pixels[8, 8]);
            SixLabors.ImageSharp.Image configured = source.ToIS;
            LargeWebpEncoding.Configure(source);
            Assert.Same(configured, source.ToIS);
        }
        finally
        {
            source._CacheISImg?.Dispose();
        }
    }
}
