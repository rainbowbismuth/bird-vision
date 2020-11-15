from birdvision.config import configure
configure()

import click
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from pathlib import Path
import cv2


@click.command()
@click.option('--clusters', default=20, help="The number of clusters")
@click.option('--full', default=False, help="Write the original image into each bucket")
def cluster_images(src, dst, clusters, full):
    """
    When manually labelling data, it's often helpful to run K-Means on it to do the bulk of the work for you.

    This function reads every .jpg in the src folder, and splits it up into clusters, writing each image into a
    folder corresponding to the cluster it found itself in, all contained with the dst folder.
    """
    src = Path(src)
    dst = Path(dst)

    k_means = MiniBatchKMeans(n_clusters=clusters)
    images = []
    for path in tqdm(list(src.glob('*.jpg'))):
        image = cv2.imread(path.as_posix())
        if image is None or image.size == 0:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        images.append((path.name, image, resized.flatten()))

    k_means.partial_fit([img for (_, _, img) in images])
    predicted = k_means.predict([img for (_, _, img) in images])

    for i, (path, orig, image) in enumerate(tqdm(images)):
        bucket = predicted[i]
        bucket_path = dst / bucket
        bucket_path.mkdir(parents=True, exist_ok=True)

        image_path = (bucket_path / f'{i:05d}_{path}').as_posix()
        if full:
            cv2.imwrite(image_path, orig)
        else:
            cv2.imwrite(image_path, image.reshape((32, 32)))
