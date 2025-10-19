# tools/debug_imports.py
import sys
print("Python path:", sys.path)

print("\n1. Testing MOSMetric import...")
try:
    from mmaction.evaluation.metrics.mos_metric import MOSMetric, MOSEvaluator
    print("✓ MOSMetric imported")
    print("✓ MOSEvaluator imported")
except Exception as e:
    print(f"✗ Import failed: {e}")

print("\n2. Testing registry...")
try:
    from mmaction.registry import METRICS
    print(f"✓ METRICS registry loaded")
    print(f"Available metrics: {list(METRICS._module_dict.keys())}")
except Exception as e:
    print(f"✗ Registry failed: {e}")

print("\n3. Testing build...")
try:
    evaluator = METRICS.build(dict(type='MOSMetric'))
    print(f"✓ MOSMetric built successfully: {type(evaluator)}")
except Exception as e:
    print(f"✗ Build failed: {e}")