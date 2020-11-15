from birdvision.config import configure
import click
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from pathlib import Path
from birdvision.node import Node
import cv2


def chunk(seq, size):
    return (seq[pos:pos+size] for pos in range(0, len(seq), size))


def load_image(path):
    image = cv2.imread(path.as_posix())
    if image is None or image.size == 0:
        return None
    return Node(image)


def for_every_image(file_chunks, f):
    for i, file_chunk in enumerate(tqdm(file_chunks)):
        images = []
        for path in file_chunk:
            image = load_image(path)
            if image is None:
                continue
            images.append((path, image))

        f(i, images)


@click.command()
@click.option('--clusters', default=20, help="The number of clusters")
@click.option('--full', 'output', flag_value='full', help="Write the original image into each bucket", default=True)
@click.option('--gray', 'output', flag_value='gray', help="Write the full grayscale image into each bucket")
@click.option('--thumb', 'output', flag_value='thumb', help="Write the thumbnail grayscale image into each bucket")
@click.argument('src')
@click.argument('dst')
def cluster_images(src, dst, clusters, output):
    """
    When manually labelling data, it's often helpful to run K-Means on it to do the bulk of the work for you.

    This function reads every image in the src folder, and splits it up into clusters, writing each image into a
    folder corresponding to the cluster it found itself in, all contained with the dst folder.
    """
    src = Path(src)
    dst = Path(dst)

    k_means = MiniBatchKMeans(n_clusters=clusters)
    file_chunks = list(chunk(list(src.glob('**/*')), size=100))
    predictions = []

    def partial_fit(_i, images):
        resized = [node.gray.thumbnail32.image.flatten() for (_, node) in images]
        k_means.partial_fit(resized)

    def predict(_i, images):
        resized = [node.gray.thumbnail32.image.flatten() for (_, node) in images]
        predictions.append(k_means.predict(resized))

    def write_buckets(i, images):
        for j, (path, node) in enumerate(images):
            # FIXME: If images fail to load the second time but not the first, predictions[i][j] will be off.
            bucket = predictions[i][j]
            bucket_path = dst / str(bucket)
            bucket_path.mkdir(parents=True, exist_ok=True)

            image_path = (bucket_path / f'{i:05d}_{path.name}').as_posix()
            if output == 'full':
                cv2.imwrite(image_path, node.image)
            elif output == 'gray':
                cv2.imwrite(image_path, node.gray.image)
            elif output == 'thumb':
                cv2.imwrite(image_path, node.gray.thumbnail32.image)
            else:
                raise Exception(f'output option "{output}" not understood')

    for_every_image(file_chunks, partial_fit)
    for_every_image(file_chunks, predict)
    for_every_image(file_chunks, write_buckets)


if __name__ == '__main__':
    configure()
    cluster_images()
