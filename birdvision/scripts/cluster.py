from pathlib import Path

import click
import cv2
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from birdvision.config import configure
from birdvision.node import Node


def chunk(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def load_image(path):
    image = cv2.imread(path.as_posix())
    if image is None or image.size == 0:
        return None
    return Node(image)


def for_every_image(file_chunks, f):
    for i, file_chunk in enumerate(tqdm(file_chunks, desc=f.__name__)):
        images = []
        for path in file_chunk:
            image = load_image(path)
            if image is None:
                continue
            images.append((path, image))

        f(i, images)


def images_for_cluster_type(images, cluster_type):
    if cluster_type == 'gray':
        return [node.gray.thumbnail32.image.flatten() for (_, node) in images]
    elif cluster_type == 'color':
        return [node.thumbnail32.image.flatten() for (_, node) in images]
    else:
        raise Exception(f'unknown cluster_type option "{cluster_type}"')


@click.command()
@click.option('--clusters', default=20, help="The number of clusters")
@click.option('--cluster-gray', 'cluster_type', flag_value='gray', help="Cluster grayscale images", default=True)
@click.option('--cluster-color', 'cluster_type', flag_value='color', help="Cluster color images")
@click.option('--full', 'output', flag_value='full', help="Write the original image into each bucket", default=True)
@click.option('--gray', 'output', flag_value='gray', help="Write the full grayscale image into each bucket")
@click.option('--thumb', 'output', flag_value='thumb', help="Write the thumbnail grayscale image into each bucket")
@click.argument('src')
@click.argument('dst')
def cluster_images(src, dst, clusters, cluster_type, output):
    """
    When manually labelling data, it's often helpful to run K-Means on it to do the bulk of the work for you.

    This function reads every image in the src folder, and splits it up into clusters, writing each image into a
    folder corresponding to the cluster it found itself in, all contained with the dst folder.
    """
    src = Path(src)
    dst = Path(dst)

    k_means = MiniBatchKMeans(n_clusters=clusters)
    file_chunks = list(chunk(list(src.glob('**/*')), size=max(100, clusters * 5)))
    predictions = []

    def partial_fit(_i, images):
        resized = images_for_cluster_type(images, cluster_type)
        k_means.partial_fit(resized)

    def predict(_i, images):
        resized = images_for_cluster_type(images, cluster_type)
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
                raise Exception(f'unknown output option "{output}"')

    for_every_image(file_chunks, partial_fit)
    for_every_image(file_chunks, predict)
    for_every_image(file_chunks, write_buckets)


if __name__ == '__main__':
    configure()
    cluster_images()
