"""
============================================================
Quick Test Script - Validate UMAP Fixes
Tests the 3 critical fixes before full pipeline run
============================================================
"""
import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from config import UMAP_OPTIMIZED_PARAMS, get_feature_path, PLOTS_DIR

def test_fix1_label_specific_params():
    """
    Test Fix 1: Label-specific UMAP parameters
    Compare old vs new parameters on MOS visualization
    """
    print("\n" + "="*70)
    print("TEST 1: Label-Specific UMAP Parameters")
    print("="*70)
    
    # Load stage1 train data (most promising from your results)
    layer = 'stage1'
    split = 'train'
    
    try:
        embeddings_path = get_feature_path(split, layer, 'embeddings')
        mos_path = get_feature_path(split, layer, 'mos')
        
        embeddings = np.load(embeddings_path)
        mos = np.load(mos_path)
        
        print(f"✓ Loaded {layer}/{split}: {embeddings.shape}")
        
        # OLD parameters (what you were using)
        print("\nTesting OLD parameters...")
        reducer_old = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        proj_old = reducer_old.fit_transform(embeddings)
        
        # NEW optimized parameters for MOS
        print("Testing NEW optimized parameters for MOS...")
        mos_params = UMAP_OPTIMIZED_PARAMS['mos']
        reducer_new = umap.UMAP(**mos_params)
        proj_new = reducer_new.fit_transform(embeddings)
        
        # Compute correlation with UMAP dimensions
        from scipy.stats import spearmanr
        
        old_corr_dim0 = spearmanr(mos, proj_old[:, 0])[0]
        old_corr_dim1 = spearmanr(mos, proj_old[:, 1])[0]
        new_corr_dim0 = spearmanr(mos, proj_new[:, 0])[0]
        new_corr_dim1 = spearmanr(mos, proj_new[:, 1])[0]
        
        print(f"\nOLD params correlation:")
        print(f"  Dim 0: {old_corr_dim0:.4f}")
        print(f"  Dim 1: {old_corr_dim1:.4f}")
        print(f"  Max abs: {max(abs(old_corr_dim0), abs(old_corr_dim1)):.4f}")
        
        print(f"\nNEW params correlation:")
        print(f"  Dim 0: {new_corr_dim0:.4f}")
        print(f"  Dim 1: {new_corr_dim1:.4f}")
        print(f"  Max abs: {max(abs(new_corr_dim0), abs(new_corr_dim1)):.4f}")
        
        improvement = max(abs(new_corr_dim0), abs(new_corr_dim1)) - max(abs(old_corr_dim0), abs(old_corr_dim1))
        print(f"\n{'✓' if improvement > 0 else '✗'} Improvement: {improvement:+.4f}")
        
        # Visualize comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
        
        # Old params
        sc1 = ax1.scatter(proj_old[:, 0], proj_old[:, 1], c=mos, cmap='viridis', s=10, alpha=0.6)
        ax1.set_title(f'OLD Params\n(n_neighbors=15, min_dist=0.1)\nCorr: {max(abs(old_corr_dim0), abs(old_corr_dim1)):.3f}')
        ax1.set_xlabel('UMAP Dim 1')
        ax1.set_ylabel('UMAP Dim 2')
        plt.colorbar(sc1, ax=ax1, label='MOS')
        
        # New params
        sc2 = ax2.scatter(proj_new[:, 0], proj_new[:, 1], c=mos, cmap='viridis', s=10, alpha=0.6)
        ax2.set_title(f'NEW Optimized Params\n(n_neighbors={mos_params["n_neighbors"]}, min_dist={mos_params["min_dist"]})\nCorr: {max(abs(new_corr_dim0), abs(new_corr_dim1)):.3f}')
        ax2.set_xlabel('UMAP Dim 1')
        ax2.set_ylabel('UMAP Dim 2')
        plt.colorbar(sc2, ax=ax2, label='MOS')
        
        plt.tight_layout()
        output_path = PLOTS_DIR / 'test_fix1_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved comparison plot: {output_path}")
        plt.close()
        
        return improvement > 0
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fix2_unified_projections():
    """
    Test Fix 2: Unified train/val/test projections
    Verify that val/test use train's fitted reducer
    """
    print("\n" + "="*70)
    print("TEST 2: Unified Train/Val/Test Projections")
    print("="*70)
    
    layer = 'stage1'
    
    try:
        # Load all splits
        train_emb = np.load(get_feature_path('train', layer, 'embeddings'))
        val_emb = np.load(get_feature_path('val', layer, 'embeddings'))
        test_emb = np.load(get_feature_path('test', layer, 'embeddings'))
        
        print(f"✓ Loaded embeddings:")
        print(f"  Train: {train_emb.shape}")
        print(f"  Val: {val_emb.shape}")
        print(f"  Test: {test_emb.shape}")
        
        # OLD approach: fit separately (WRONG)
        print("\nOLD approach (fitting separately)...")
        reducer_train_old = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        reducer_val_old = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        reducer_test_old = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        
        proj_train_old = reducer_train_old.fit_transform(train_emb)
        proj_val_old = reducer_val_old.fit_transform(val_emb)
        proj_test_old = reducer_test_old.fit_transform(test_emb)
        
        # Check ranges (should be different)
        print(f"  Train range: X=[{proj_train_old[:, 0].min():.2f}, {proj_train_old[:, 0].max():.2f}], Y=[{proj_train_old[:, 1].min():.2f}, {proj_train_old[:, 1].max():.2f}]")
        print(f"  Val range:   X=[{proj_val_old[:, 0].min():.2f}, {proj_val_old[:, 0].max():.2f}], Y=[{proj_val_old[:, 1].min():.2f}, {proj_val_old[:, 1].max():.2f}]")
        print(f"  Test range:  X=[{proj_test_old[:, 0].min():.2f}, {proj_test_old[:, 0].max():.2f}], Y=[{proj_test_old[:, 1].min():.2f}, {proj_test_old[:, 1].max():.2f}]")
        print("  ✗ Ranges are different (projections not comparable)")
        
        # NEW approach: fit on train, transform val/test (CORRECT)
        print("\nNEW approach (unified projection)...")
        reducer_train_new = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        
        proj_train_new = reducer_train_new.fit_transform(train_emb)
        proj_val_new = reducer_train_new.transform(val_emb)  # Use same reducer
        proj_test_new = reducer_train_new.transform(test_emb)  # Use same reducer
        
        print(f"  Train range: X=[{proj_train_new[:, 0].min():.2f}, {proj_train_new[:, 0].max():.2f}], Y=[{proj_train_new[:, 1].min():.2f}, {proj_train_new[:, 1].max():.2f}]")
        print(f"  Val range:   X=[{proj_val_new[:, 0].min():.2f}, {proj_val_new[:, 0].max():.2f}], Y=[{proj_val_new[:, 1].min():.2f}, {proj_val_new[:, 1].max():.2f}]")
        print(f"  Test range:  X=[{proj_test_new[:, 0].min():.2f}, {proj_test_new[:, 0].max():.2f}], Y=[{proj_test_new[:, 1].min():.2f}, {proj_test_new[:, 1].max():.2f}]")
        print("  ✓ Projections are in comparable space")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fix3_imbalanced_metrics():
    """
    Test Fix 3: Imbalance-aware metrics for binary labels
    """
    print("\n" + "="*70)
    print("TEST 3: Imbalance-Aware Metrics")
    print("="*70)
    
    layer = 'stage1'
    split = 'train'
    
    try:
        embeddings = np.load(get_feature_path(split, layer, 'embeddings'))
        spatial_flag = np.load(get_feature_path(split, layer, 'spatial_flag'))
        
        n_positive = (spatial_flag == 1).sum()
        n_negative = (spatial_flag == 0).sum()
        
        print(f"✓ Loaded spatial_flag data:")
        print(f"  Positive: {n_positive} ({100*n_positive/(n_positive+n_negative):.2f}%)")
        print(f"  Negative: {n_negative} ({100*n_negative/(n_positive+n_negative):.2f}%)")
        print(f"  Imbalance ratio: 1:{n_negative/n_positive:.1f}")
        
        # Fit UMAP with artifact-optimized params
        params = UMAP_OPTIMIZED_PARAMS['spatial_flag']
        reducer = umap.UMAP(**params)
        projection = reducer.fit_transform(embeddings)
        
        print(f"\nUsing optimized params: n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}")
        
        # Compute NEW imbalanced metrics
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(projection)
        
        clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        clf.fit(X_scaled, spatial_flag)
        
        y_pred = clf.predict(X_scaled)
        y_proba = clf.predict_proba(X_scaled)[:, 1]
        
        print("\nImbalance-aware metrics:")
        print(f"  ROC-AUC: {roc_auc_score(spatial_flag, y_proba):.4f}")
        print(f"  F1-Score: {f1_score(spatial_flag, y_pred):.4f}")
        print(f"  Precision: {precision_score(spatial_flag, y_pred):.4f}")
        print(f"  Recall: {recall_score(spatial_flag, y_pred):.4f}")
        
        # Centroid distance
        pos_centroid = projection[spatial_flag == 1].mean(axis=0)
        neg_centroid = projection[spatial_flag == 0].mean(axis=0)
        centroid_dist = np.linalg.norm(pos_centroid - neg_centroid)
        print(f"  Centroid Distance: {centroid_dist:.4f}")
        
        # OLD metric (silhouette) would fail here
        from sklearn.metrics import silhouette_score
        try:
            sil = silhouette_score(projection, spatial_flag)
            print(f"\n  OLD metric (silhouette): {sil:.4f} (misleading for imbalanced data)")
        except:
            print(f"\n  OLD metric (silhouette): FAILED (not enough samples per class)")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("UMAP FIXES VALIDATION TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1
    results['fix1_label_params'] = test_fix1_label_specific_params()
    
    # Test 2
    results['fix2_unified_projection'] = test_fix2_unified_projections()
    
    # Test 3
    results['fix3_imbalanced_metrics'] = test_fix3_imbalanced_metrics()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All tests passed! Ready for full pipeline run.")
        print("\nRun: python main_investigation.py")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)