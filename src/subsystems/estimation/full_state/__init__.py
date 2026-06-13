"""Full-state estimators over [x, y, vx, vy, theta, omega].

Two interchangeable backends implement ``FullStateEstimator``:
the CUDA ``ParticleFilter`` and the filterpy-based ``FullStateKF``.
"""
