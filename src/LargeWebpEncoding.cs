using System.Buffers;
using System.Runtime.CompilerServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Memory;
using SixLabors.ImageSharp.PixelFormats;
using SwarmUI.Media;
using SwarmUI.Utils;

namespace SeedVRTiling;

/// <summary>Applies the large-buffer allocator to an individual generated WebP image.</summary>
internal static class LargeWebpEncoding
{
    private sealed class ConfiguredClone(Configuration configuration) : IImageVisitor
    {
        internal SixLabors.ImageSharp.Image Result;

        public void Visit<TPixel>(Image<TPixel> image) where TPixel : unmanaged, IPixel<TPixel>
            => Result = image.Clone(configuration);
    }

    internal static bool NeedsLargeBuffers(string format, int width, int height)
        => format.StartsWith("WEBP", StringComparison.OrdinalIgnoreCase)
            && width <= OutputFormatGuard.WebpMaxDimension && height <= OutputFormatGuard.WebpMaxDimension
            && (long)width * height * 8 > LargeWebpAllocator.DefaultBufferLimit;

    internal static void Prepare(ImageFile file, string format, int width, int height)
    {
        if (!NeedsLargeBuffers(format, width, height))
        {
            return;
        }
        Configure(file);
        Logs.Debug($"[SeedVR Tiling] Enabling large WebP encoder buffers for {width}x{height}; output remains WebP.");
    }

    internal static void Configure(ImageFile file, MemoryAllocator allocator = null)
    {
        lock (file)
        {
            SixLabors.ImageSharp.Image cached = file._CacheISImg;
            if (cached?.Configuration.MemoryAllocator is LargeWebpAllocator)
            {
                return;
            }
            Configuration configuration = (cached?.Configuration ?? Configuration.Default).Clone();
            configuration.MemoryAllocator = allocator ?? new LargeWebpAllocator(configuration.MemoryAllocator);
            // Keep the allocator private to this image, including stealth-alpha encoding.
            SixLabors.ImageSharp.Image configured;
            if (cached is null)
            {
                configured = SixLabors.ImageSharp.Image.Load(new DecoderOptions { Configuration = configuration }, file.RawData);
            }
            else
            {
                ConfiguredClone clone = new(configuration);
                cached.AcceptVisitor(clone);
                configured = clone.Result;
            }
            file._CacheISImg = configured;
            cached?.Dispose();
        }
    }
}

/// <summary>
/// Retains ImageSharp's normal pooling while allowing large WebP scratch buffers.
/// ImageSharp 3.1.x caps individual default-allocator buffers at 1 GiB even when
/// AllocationLimitMegabytes is raised. WebP's lossless/alpha encoder requests
/// eight bytes per pixel in one buffer, which can exceed that cap at legal sizes.
/// </summary>
internal sealed class LargeWebpAllocator(MemoryAllocator pooled, long pooledLimitBytes = LargeWebpAllocator.DefaultBufferLimit) : MemoryAllocator
{
    internal const long DefaultBufferLimit = 1L << 30;

    // Keep image storage in ordinary pooled chunks. Only large contiguous
    // encoder allocations take the managed-array path below.
    protected override int GetBufferCapacityInBytes() => 4 * 1024 * 1024;

    public override IMemoryOwner<T> Allocate<T>(int length, AllocationOptions options = AllocationOptions.None)
    {
        long bytes = (long)length * Unsafe.SizeOf<T>();
        if (bytes <= pooledLimitBytes || bytes > int.MaxValue)
        {
            return pooled.Allocate<T>(length, options);
        }
        return new LargeBuffer<T>(length, options);
    }

    private sealed class LargeBuffer<T> : IMemoryOwner<T> where T : struct
    {
        private T[] data;

        internal LargeBuffer(int length, AllocationOptions options)
        {
            data = (options & AllocationOptions.Clean) != 0
                ? new T[length] : GC.AllocateUninitializedArray<T>(length);
        }

        public Memory<T> Memory => data ?? throw new ObjectDisposedException(nameof(LargeBuffer<T>));

        public void Dispose() => data = null;
    }
}
