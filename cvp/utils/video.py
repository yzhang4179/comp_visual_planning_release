import imageio


def save_mp4(images, filename, fps=10):
    """Write a list of (H, W, 3) uint8 images to an mp4 file."""
    writer = imageio.get_writer(filename, fps=fps, codec="libx264", quality=10, pixelformat="yuv420p")
    for image in images:
        writer.append_data(image)
    writer.close()
