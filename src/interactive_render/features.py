from skimage import feature
from skimage.measure import ransac
from skimage.transform import (
    EuclideanTransform,
    SimilarityTransform,
    AffineTransform
)


# default SIFT parameters
PARAMS_SIFT = {
    "upsampling": 1,  # no upsampling
    "n_octaves": 3,
    "sigma_min": 3
}

# default ORB parameters
PARAMS_ORB = {
    "downscale": 1.2,  # downscale factor for image pyramid
    "n_scales": 8
}

# default `match_descriptors` parameters
PARAMS_MATCH = {
    "metric": None,
    "cross_check": True,
    "max_ratio": 0.8
}

# default RANSAC parameters
PARAMS_RANSAC = {
    "model_class": EuclideanTransform,
    "min_samples": 12,
    "residual_threshold": 2,
    "max_trials": 10000
}


def find_SIFT_features(
    image,
    params_SIFT=None,
):
    """Find SIFT features within an image."""
    # parameter handling
    params_SIFT = {} if params_SIFT is None else params_SIFT
    params_SIFT = {
        **PARAMS_SIFT,
        **params_SIFT
    }
    # initialize SIFT instance
    sift = feature.SIFT(**params_SIFT)
    # detect and extract features
    sift.detect_and_extract(image)
    return sift


def find_ORB_features(
    image,
    params_ORB=None
):
    """Find ORB features within an image."""
    # parameter handling
    params_ORB = {} if params_ORB is None else params_ORB
    params_ORB = {
        **PARAMS_ORB,
        **params_ORB
    }
    # initialize ORB instance
    orb = feature.ORB(**params_ORB)
    # detect and extract features
    orb.detect_and_extract(image)
    return orb


def match_features(
    features_p,
    features_q,
    params_match=None
    ):
    """Applies brute-force matching of feature descriptors.

    Parameters
    ----------
    features_p : 
    features_q : 
    params_match : dict (optional)
        parameters for brute-force descriptor matching

    Returns
    -------
    matches_p : (2, N) float array
    matches_q : (2, N) float array
    """
    # parameter handling
    params_match = {} if params_match is None else params_match
    params_match = {
        **PARAMS_MATCH,
        **params_match
    }

    # brute-force matching of descriptors
    matches = feature.match_descriptors(
        features_p.descriptors,
        features_q.descriptors,
        **params_match
    )

    # extract matched keypoints
    matches_p = features_p.keypoints[matches[:, 0]]
    matches_q = features_q.keypoints[matches[:, 1]]

    return matches_p[:, [1, 0]], matches_q[:, [1, 0]]


def find_feature_correspondences(
    image_p,
    image_q,
    feature_detector="SIFT",
    params_features=None,
    params_match=None
):
    """Find pairwise feature correspondences between two images."""
    if feature_detector == "SIFT":
        feature_finder = find_SIFT_features
    elif feature_detector == "ORB":
        feature_finder = find_ORB_features
    else:
        msg = f"'{feature_detector}' feature detector not implemented."
        raise ValueError(msg)
    # find features with whatever algorithm
    features_p = feature_finder(image_p, params_features)
    features_q = feature_finder(image_q, params_features)

    # brute-force descriptor matching
    matches_p, matches_q = match_features(
        features_p,
        features_q,
        params_match
    )

    return matches_p, matches_q


def find_robust_feature_correspondences(
    image_p,
    image_q,
    feature_detector="SIFT",
    params_features=None,
    params_match=None,
    params_RANSAC=None,
):
    """Find robust, pairwise feature correspondences between two images.

    Features are detected and extracted from the provided feature detector
    algorithm. Brute-force matching of feature descriptors is applied by
    default. RANSAC is run on matched feature positions to enure robustness,
    aka that coordinates of feature correspondences can be modeled by some
    linear transformation.

    Parameters
    ----------
    image_p : (M, N) ubyte array
    image_q : (M, N) ubyte array
    feature_detector: str
    params_features : dict
        parameter dict passed to chosen `feature_detector`
    params_match : dict
        parameter dict passed to `match_descriptors`
    params_RANSAC : dict
        parameter dict passed to `ransac`

    Returns
    -------
    matches_p : (N, 2) int64 array
    matches_q : (N, 2) int64 array
    """
    # parameter handling
    params_RANSAC = {} if params_RANSAC is None else params_RANSAC
    params_RANSAC = {
        **PARAMS_RANSAC,
        **params_RANSAC
    }

    # find (semi-robust) features
    matches_p, matches_q = find_feature_correspondences(
        image_p,
        image_q,
        feature_detector,
        params_features,
        params_match,
    )

    # robustify the feature correspondences
    model, inliers = ransac(
        (matches_p, matches_q),
        **params_RANSAC
    )

    # TODO: some kind of error checking based on returned transformation

    return matches_p[inliers], matches_q[inliers]
