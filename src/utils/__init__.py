# utils/__init__.py
# Re-exporta os símbolos principais para facilitar imports nos scripts.

from utils.config import (
    BASE_PATH, DATA_TRAIN, DATA_TEST, PRETRAINED,
    KPCONV_WEIGHTS, PTRANSF_WEIGHTS,
    INPUT_DIM, D_MODEL, NUM_HEADS, NUM_LAYERS,
    VOXEL_SIZE, ANOMALY_PCTL,
    setup_logging, detect_environment,
    # bibliotecas
    os, gc, math, time, json, random, logging, warnings, psutil,
    deque, datetime, Path,
    np, o3d, PlyData, cKDTree, Rotation,
    plt, sns, pd,
    torch, nn, F, Dataset, DataLoader, grad_checkpoint,
    spectral_norm, AdamW, autocast, GradScaler,
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    jaccard_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    IsolationForest, OneClassSVM, LocalOutlierFactor,
)

from utils.data import (
    PointCloudPreprocessor, PointCloudAugmentation,
    PointCloudDataset, collate_fn,
    load_ply_file, load_folder, split_dataset, make_loaders,
    PREPROCESSOR,
)

from utils.building_blocks import (
    LightweightGraphConv, LocalSpatialAttention, MultiScaleAggregation,
    DensityAwareNorm, GatedResidualConnection, _selective_load,
)

from utils.architectures import (
    KPFCNNInspiredAdvanced, GANGenerator,
    SpatialPositionalEncoding3D, SpatialMultiHeadSelfAttention,
    GatedFeedForward, PointTransformerBlock, LocalFeatureAggregation,
    PointTransformerInspiredAdvanced, GANDiscriminator,
)

from utils.training_utils import (
    SmartEarlyStopping, compute_gradient_penalty,
    MultiScaleReconLoss, NormalMemoryBank,
    check_training_health, adaptive_lambda_recon,
    PushPullLoss,
)

from utils.evaluation import (
    fit_gmm_threshold, apply_threshold, evaluate, save_results, plot_training_history,
    plot_score_distribution, chamfer_distance,
    statistical_comparison,
    compute_severity, severity_report,
    classify_crack,
    calibrate_threshold_f1, calibrate_threshold_gmean, calibrate_threshold_fbeta,
    apply_spatial_coherence,
    compute_crack_sf_interval, apply_scalar_field_gate,
    visualize_cracks,
    CRACK_CATEGORIES, ABNT_LIMITS_MM, CAA_PROJETO,
    ScalarFieldGMM, save_colored_ply,
    evaluate_ablation, compare_models,
)

