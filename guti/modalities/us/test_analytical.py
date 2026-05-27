# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from guti.modalities.us.pat_analytical_integrated import (
    analytical_solver, 
    get_detector_angles, 
    get_source_params,
    R_brain, R_skull, R_scalp,
    alpha_brain, beta_skull, h_skull
)

from guti.core import get_source_positions, get_sensor_positions_spiral

# %%
# Setup geometry and parameters using utils functions
print("Testing analytical ultrasound model...")
print(f"Geometry: brain={R_brain}mm, skull={R_skull}mm, scalp={R_scalp}mm")
print(f"Skull thickness: {h_skull}mm")
print(f"Attenuation: brain={alpha_brain:.4f} Np/mm, skull={beta_skull:.4f} Np/mm")

# %%
# Create source and detector positions using core functions
print("\nCreating sources and detectors using core functions...")

# Get source positions (inside brain)
print("Creating sources...")
source_positions = get_source_positions() - np.array([R_brain, R_brain, 0])

# Get detector positions (on scalp surface)  
print("Creating detectors...")
n_test_detectors = 32  # Use fewer detectors for testing
detector_positions = get_sensor_positions_spiral(n_sensors=n_test_detectors, offset=10) - np.array([R_scalp, R_scalp, 0])

print(f"Number of sources: {len(source_positions)}")
print(f"Number of detectors: {len(detector_positions)}")

# Show a few examples
print(f"\nFirst 5 source positions (mm):")
for i in range(min(5, len(source_positions))):
    pos = source_positions[i]
    print(f"  Source {i+1}: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

print(f"\nFirst 8 detector positions (mm):")
for i in range(min(8, len(detector_positions))):
    pos = detector_positions[i]
    print(f"  Detector {i+1}: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

# %%
# Use centered coordinate system (matching pat_analytical_integrated.py)
print("\nUsing centered coordinate system...")

# With the new positioning method, we use center at origin [0, 0, 0]
center_mm = np.array([0, 0, 0])

# Debug: Check current positions
source_center = np.mean(source_positions, axis=0)  
detector_center = np.mean(detector_positions, axis=0)

print(f"Using center: [{center_mm[0]:.1f}, {center_mm[1]:.1f}, {center_mm[2]:.1f}] mm")
print(f"Source positions center: [{source_center[0]:.1f}, {source_center[1]:.1f}, {source_center[2]:.1f}] mm")
print(f"Detector positions center: [{detector_center[0]:.1f}, {detector_center[1]:.1f}, {detector_center[2]:.1f}] mm")

# Verify distances make sense
source_distances = np.linalg.norm(source_positions - center_mm, axis=1)
detector_distances = np.linalg.norm(detector_positions - center_mm, axis=1)

print(f"Source distances from center: [{np.min(source_distances):.1f}, {np.max(source_distances):.1f}] mm (should be < {R_brain})")
print(f"Detector distances from center: [{np.min(detector_distances):.1f}, {np.max(detector_distances):.1f}] mm")

# %%
# Test coordinate conversions on a subset for display
print("\nTesting coordinate conversions...")
gamma_det, phi_det = get_detector_angles(detector_positions, center_mm)
R_sources, source_angles = get_source_params(source_positions, center_mm)

print("First 8 detector angles:")
for i in range(min(8, len(detector_positions))):
    print(f"  Detector {i+1}: gamma={gamma_det[i]:.3f} rad, phi={phi_det[i]:.3f} rad")

print("First 5 source parameters:")
for i in range(min(5, len(source_positions))):
    print(f"  Source {i+1}: R={R_sources[i]:.3f} mm, angle={source_angles[i]:.3f} rad")

# %%
# Compute analytical signals (use subset for testing)
print("\nComputing analytical signals...")
# Use smaller subset for testing to avoid long computation times
n_test_sources = min(10, len(source_positions))
n_test_detectors = min(16, len(detector_positions))

test_source_positions = source_positions[:n_test_sources]
test_detector_positions = detector_positions[:n_test_detectors]

print(f"Using {n_test_sources} sources and {n_test_detectors} detectors for testing")

signals = analytical_solver(test_source_positions, test_detector_positions, center_mm)

print(f"Signal matrix shape: {signals.shape}")
print(f"Signal range: [{np.min(signals):.6f}, {np.max(signals):.6f}]")
print(f"Signal range (log): [{np.log10(np.max([np.min(signals), 1e-10])):.2f}, {np.log10(np.max(signals)):.2f}]")

# %%
# Plot 1: Signal matrix (logarithmic scale)
plt.figure(figsize=(8, 6))
# Use logarithmic scale, avoiding log(0) by adding small epsilon
signals_log = np.log10(np.maximum(signals, 1e-10))
plt.imshow(signals_log, aspect='auto', cmap='viridis')
plt.colorbar(label='Signal strength (log10)')
plt.xlabel('Source index')
plt.ylabel('Detector index')
plt.title('Analytical Signal Matrix (Log Scale)')
plt.tight_layout()
plt.show()

# %%
# Plot 2: Signals for first source
plt.figure(figsize=(8, 6))
plt.plot(signals[:, 0], 'o-')
plt.xlabel('Detector index')
plt.ylabel('Signal strength')
plt.title(f'Signals from Source 1')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Plot 3: Signals at first detector
plt.figure(figsize=(8, 6))
plt.plot(signals[0, :], 's-')
plt.xlabel('Source index')
plt.ylabel('Signal strength')
plt.title(f'Signals at Detector 1')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Plot 4: 3D visualization of geometry
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot test sources (subset used in computation)
ax.scatter(test_source_positions[:, 0], test_source_positions[:, 1], test_source_positions[:, 2], 
           c='red', s=100, label=f'Test Sources ({n_test_sources})', marker='o')

# Plot test detectors (subset used in computation)
ax.scatter(test_detector_positions[:, 0], test_detector_positions[:, 1], test_detector_positions[:, 2], 
           c='blue', s=100, label=f'Test Detectors ({n_test_detectors})', marker='^')

# Plot all sources (smaller, translucent)
ax.scatter(source_positions[:, 0], source_positions[:, 1], source_positions[:, 2], 
           c='red', s=20, alpha=0.3, label=f'All Sources ({len(source_positions)})', marker='.')

# Plot all detectors (smaller, translucent)  
ax.scatter(detector_positions[:, 0], detector_positions[:, 1], detector_positions[:, 2], 
           c='blue', s=20, alpha=0.3, label=f'All Detectors ({len(detector_positions)})', marker='.')

# Plot analytical center (used for calculations)
ax.scatter(*center_mm, c='black', s=100, label='Analytical Center', marker='x')

# Draw brain and skull spheres (centered at origin)
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)  # full sphere

# Brain surface (centered at origin)
x_brain = R_brain * np.outer(np.cos(u), np.sin(v)) + center_mm[0]
y_brain = R_brain * np.outer(np.sin(u), np.sin(v)) + center_mm[1]
z_brain = R_brain * np.outer(np.ones(np.size(u)), np.cos(v)) + center_mm[2]
ax.plot_wireframe(x_brain, y_brain, z_brain, alpha=0.2, color='red', linewidth=0.5)

# Skull surface (centered at origin)
x_skull = R_skull * np.outer(np.cos(u), np.sin(v)) + center_mm[0]
y_skull = R_skull * np.outer(np.sin(u), np.sin(v)) + center_mm[1]
z_skull = R_skull * np.outer(np.ones(np.size(u)), np.cos(v)) + center_mm[2]
ax.plot_wireframe(x_skull, y_skull, z_skull, alpha=0.1, color='gray', linewidth=0.5)

# Scalp surface (centered at origin)
x_scalp = R_scalp * np.outer(np.cos(u), np.sin(v)) + center_mm[0]
y_scalp = R_scalp * np.outer(np.sin(u), np.sin(v)) + center_mm[1]
z_scalp = R_scalp * np.outer(np.ones(np.size(u)), np.cos(v)) + center_mm[2]
ax.plot_wireframe(x_scalp, y_scalp, z_scalp, alpha=0.1, color='blue', linewidth=0.5)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Source-Detector Geometry')
ax.legend()

plt.tight_layout()
plt.show()

print("\nTest completed successfully!") 
# %%

# %%
# Plot integral as function of gamma and phi angles
def plot_integral_vs_angles(source_pos, center_mm, n_gamma=30, n_phi=30):
    """Plot the analytical integral as a function of detector angles gamma and phi"""
    from guti.modalities.us.pat_analytical_integrated import compute_analytical_response, get_source_params
    
    # Use first source for this analysis
    if len(source_pos) > 0:
        test_source = source_pos[0]
        print(f"Analyzing integral for source at: [{test_source[0]:.1f}, {test_source[1]:.1f}, {test_source[2]:.1f}] mm")
        
        # Get source parameters
        R_src, source_angle = get_source_params(test_source.reshape(1, -1), center_mm)
        R_src = R_src[0]
        source_angle = source_angle[0]
        
        print(f"Source parameters: R={R_src:.2f} mm, angle={source_angle:.4f} rad ({np.degrees(source_angle):.1f}°)")
        
        # Create gamma and phi grids
        gamma_range = np.linspace(-np.pi/3, np.pi/3, n_gamma)  # ±60 degrees
        phi_range = np.linspace(-np.pi, np.pi, n_phi)         # Full circle
        
        Gamma, Phi = np.meshgrid(gamma_range, phi_range, indexing='ij')
        
        # Compute integral for each (gamma, phi) pair
        integral_values = np.zeros_like(Gamma)
        
        print("Computing integral over gamma-phi grid...")
        for i in range(n_gamma):
            for j in range(n_phi):
                gamma_val = Gamma[i, j]
                phi_val = Phi[i, j]
                
                # Compute integral for this detector angle
                integral_val = compute_analytical_response(gamma_val, phi_val, R_src, source_angle)
                integral_values[i, j] = float(integral_val)
            
            if i % 5 == 0:
                print(f"  Processed gamma step {i+1}/{n_gamma}")
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Linear scale
        ax1 = axes[0, 0]
        im1 = ax1.contourf(np.degrees(Phi), np.degrees(Gamma), integral_values, levels=20, cmap='viridis')
        ax1.set_xlabel('Phi (degrees)')
        ax1.set_ylabel('Gamma (degrees)')
        ax1.set_title('Integral vs Detector Angles (Linear Scale)')
        plt.colorbar(im1, ax=ax1, label='Integral Value')
        
        # Plot 2: Log scale
        ax2 = axes[0, 1]
        integral_positive = np.maximum(integral_values, 1e-12)
        im2 = ax2.contourf(np.degrees(Phi), np.degrees(Gamma), np.log10(integral_positive), levels=20, cmap='viridis')
        ax2.set_xlabel('Phi (degrees)')
        ax2.set_ylabel('Gamma (degrees)')
        ax2.set_title('Integral vs Detector Angles (Log10 Scale)')
        plt.colorbar(im2, ax=ax2, label='Log10 Integral Value')
        
        # Plot 3: Cross-section at phi=0
        ax3 = axes[1, 0]
        phi_zero_idx = n_phi // 2  # Middle index corresponds to phi=0
        ax3.plot(np.degrees(gamma_range), integral_values[:, phi_zero_idx], 'b-', linewidth=2)
        ax3.set_xlabel('Gamma (degrees)')
        ax3.set_ylabel('Integral Value')
        ax3.set_title('Cross-section at Phi = 0°')
        ax3.grid(True)
        
        # Plot 4: Cross-section at gamma=0
        ax4 = axes[1, 1]
        gamma_zero_idx = n_gamma // 2  # Middle index corresponds to gamma=0
        ax4.plot(np.degrees(phi_range), integral_values[gamma_zero_idx, :], 'r-', linewidth=2)
        ax4.set_xlabel('Phi (degrees)')
        ax4.set_ylabel('Integral Value')
        ax4.set_title('Cross-section at Gamma = 0°')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nIntegral statistics:")
        print(f"Min value: {np.min(integral_values):.2e}")
        print(f"Max value: {np.max(integral_values):.2e}")
        print(f"Mean value: {np.mean(integral_values):.2e}")
        print(f"Std value: {np.std(integral_values):.2e}")
        print(f"Number of infinite values: {np.sum(np.isinf(integral_values))}")
        print(f"Number of NaN values: {np.sum(np.isnan(integral_values))}")
        
        return Gamma, Phi, integral_values
    else:
        print("No source positions available!")
        return None, None, None

# Call the function
print("\n" + "="*60)
print("PLOTTING INTEGRAL VS DETECTOR ANGLES")
print("="*60)
Gamma, Phi, integral_vals = plot_integral_vs_angles(source_positions, center_mm)

# %%

# %%
# Debug: Compare parameter ranges between gamma-phi plot and real source-detector pairs
def debug_parameter_ranges(source_positions, detector_positions, center_mm):
    """Compare parameter ranges between controlled gamma-phi plot and real source-detector calculations"""
    print("\nDEBUGGING PARAMETER RANGES")
    print("="*60)
    
    # 1. Show ranges used in gamma-phi plot
    print("1. CONTROLLED RANGES (gamma-phi plot):")
    gamma_plot_range = np.linspace(-np.pi/3, np.pi/3, 30)
    phi_plot_range = np.linspace(-np.pi, np.pi, 30)
    
    print(f"   Gamma range: [{np.degrees(gamma_plot_range[0]):.1f}°, {np.degrees(gamma_plot_range[-1]):.1f}°]")
    print(f"   Phi range: [{np.degrees(phi_plot_range[0]):.1f}°, {np.degrees(phi_plot_range[-1]):.1f}°]")
    
    # Get source parameters for first source (used in gamma-phi plot)
    test_source = source_positions[0]
    R_src_plot, source_angle_plot = get_source_params(test_source.reshape(1, -1), center_mm)
    print(f"   R_src (plot): {R_src_plot[0]:.2f} mm")
    print(f"   source_angle (plot): {source_angle_plot[0]:.4f} rad ({np.degrees(source_angle_plot[0]):.1f}°)")
    
    # 2. Show ranges from real source-detector calculations
    print("\n2. REAL RANGES (from actual positions):")
    
    # Compute all detector angles
    gamma_real, phi_real = get_detector_angles(detector_positions, center_mm)
    print(f"   Gamma range: [{np.degrees(np.min(gamma_real)):.1f}°, {np.degrees(np.max(gamma_real)):.1f}°]")
    print(f"   Phi range: [{np.degrees(np.min(phi_real)):.1f}°, {np.degrees(np.max(phi_real)):.1f}°]")
    
    # Compute all source parameters
    R_real, source_angles_real = get_source_params(source_positions, center_mm)
    print(f"   R_src range: [{np.min(R_real):.2f}, {np.max(R_real):.2f}] mm")
    print(f"   source_angle range: [{np.min(source_angles_real):.4f}, {np.max(source_angles_real):.4f}] rad")
    print(f"   source_angle range: [{np.degrees(np.min(source_angles_real)):.1f}°, {np.degrees(np.max(source_angles_real)):.1f}°]")
    
    # 3. Check for extreme values
    print("\n3. CHECKING FOR EXTREME VALUES:")
    
    # Check for gamma/phi outside plot ranges
    gamma_outside = np.sum((gamma_real < -np.pi/3) | (gamma_real > np.pi/3))
    phi_outside = np.sum((phi_real < -np.pi) | (phi_real > np.pi))
    print(f"   Detectors with gamma outside plot range: {gamma_outside}/{len(gamma_real)}")
    print(f"   Detectors with phi outside plot range: {phi_outside}/{len(phi_real)}")
    
    # Check for extreme source angles (near ±90°)
    extreme_angles = np.sum(np.abs(np.abs(source_angles_real) - np.pi/2) < 0.1)
    print(f"   Sources with extreme angles (near ±90°): {extreme_angles}/{len(source_angles_real)}")
    
    # Check for very small R values
    small_R = np.sum(R_real < 1.0)
    print(f"   Sources with very small R (< 1mm): {small_R}/{len(R_real)}")
    
    # 4. Test a few extreme combinations
    print("\n4. TESTING EXTREME PARAMETER COMBINATIONS:")
    
    from guti.modalities.us.pat_analytical_integrated import compute_analytical_response
    
    # Find most extreme values
    extreme_gamma_idx = np.argmax(np.abs(gamma_real))
    extreme_phi_idx = np.argmax(np.abs(phi_real))
    extreme_angle_idx = np.argmax(np.abs(np.abs(source_angles_real) - np.pi/2))
    
    print(f"   Most extreme gamma: {np.degrees(gamma_real[extreme_gamma_idx]):.1f}° (detector {extreme_gamma_idx})")
    print(f"   Most extreme phi: {np.degrees(phi_real[extreme_phi_idx]):.1f}° (detector {extreme_phi_idx})")
    print(f"   Most extreme source angle: {np.degrees(source_angles_real[extreme_angle_idx]):.1f}° (source {extreme_angle_idx})")
    
    # Test these extreme combinations
    test_cases = [
        ("extreme_gamma", gamma_real[extreme_gamma_idx], phi_real[0], R_real[0], source_angles_real[0]),
        ("extreme_phi", gamma_real[0], phi_real[extreme_phi_idx], R_real[0], source_angles_real[0]),
        ("extreme_source_angle", gamma_real[0], phi_real[0], R_real[extreme_angle_idx], source_angles_real[extreme_angle_idx]),
        ("normal_case", gamma_real[0], phi_real[0], R_real[0], source_angles_real[0])
    ]
    
    for case_name, gamma, phi, R, src_angle in test_cases:
        try:
            result = compute_analytical_response(gamma, phi, R, src_angle)
            status = "finite" if np.isfinite(result) else ("inf" if np.isinf(result) else "nan")
            print(f"   {case_name}: {float(result):.2e} ({status})")
        except Exception as e:
            print(f"   {case_name}: ERROR - {e}")
    
    # 5. Show histograms of parameter distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Gamma distribution
    axes[0,0].hist(np.degrees(gamma_real), bins=20, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(np.degrees(-np.pi/3), color='red', linestyle='--', label='Plot range')
    axes[0,0].axvline(np.degrees(np.pi/3), color='red', linestyle='--')
    axes[0,0].set_xlabel('Gamma (degrees)')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('Real Gamma Distribution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Phi distribution
    axes[0,1].hist(np.degrees(phi_real), bins=20, alpha=0.7, edgecolor='black')
    axes[0,1].axvline(-180, color='red', linestyle='--', label='Plot range')
    axes[0,1].axvline(180, color='red', linestyle='--')
    axes[0,1].set_xlabel('Phi (degrees)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Real Phi Distribution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # R distribution
    axes[0,2].hist(R_real, bins=20, alpha=0.7, edgecolor='black')
    axes[0,2].set_xlabel('R (mm)')
    axes[0,2].set_ylabel('Count')
    axes[0,2].set_title('Real R Distribution')
    axes[0,2].grid(True, alpha=0.3)
    
    # Source angle distribution
    axes[1,0].hist(np.degrees(source_angles_real), bins=20, alpha=0.7, edgecolor='black')
    axes[1,0].axvline(-90, color='red', linestyle='--', label='±90° (tan singularity)')
    axes[1,0].axvline(90, color='red', linestyle='--')
    axes[1,0].set_xlabel('Source Angle (degrees)')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Real Source Angle Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Distance distribution (source to center)
    source_distances = np.linalg.norm(source_positions, axis=1)
    axes[1,1].hist(source_distances, bins=20, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(R_brain, color='red', linestyle='--', label=f'Brain radius ({R_brain}mm)')
    axes[1,1].set_xlabel('Source Distance from Center (mm)')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Source Distance Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Detector distance distribution
    detector_distances = np.linalg.norm(detector_positions, axis=1)
    axes[1,2].hist(detector_distances, bins=20, alpha=0.7, edgecolor='black')
    axes[1,2].axvline(R_scalp, color='blue', linestyle='--', label=f'Scalp radius ({R_scalp}mm)')
    axes[1,2].set_xlabel('Detector Distance from Center (mm)')
    axes[1,2].set_ylabel('Count')
    axes[1,2].set_title('Detector Distance Distribution')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # 6. Test gamma values systematically to confirm the threshold
    print("\n5. SYSTEMATIC GAMMA TESTING:")
    
    # Use the extreme detector that caused infinity
    extreme_gamma_idx = np.argmax(np.abs(gamma_real))
    test_phi = phi_real[0]  # Use a normal phi value
    test_R = R_real[0]      # Use a normal R value  
    test_src_angle = source_angles_real[0]  # Use a normal source angle
    
    print(f"   Testing gamma values with fixed params: phi={np.degrees(test_phi):.1f}°, R={test_R:.1f}mm, src_angle={np.degrees(test_src_angle):.1f}°")
    
    # Test a range of gamma values to find the threshold
    gamma_test_values = np.linspace(0, np.pi/2 - 0.01, 20)  # 0° to 89°
    
    from guti.modalities.us.pat_analytical_integrated import compute_analytical_response
    
    print(f"   {'Gamma (°)':<10} {'Result':<15} {'Status':<8}")
    print("   " + "-" * 35)
    
    for gamma_test in gamma_test_values:
        try:
            result = compute_analytical_response(gamma_test, test_phi, test_R, test_src_angle)
            status = "finite" if np.isfinite(result) else ("inf" if np.isinf(result) else "nan")
            result_str = f"{float(result):.2e}" if np.isfinite(result) else str(float(result))
            print(f"   {np.degrees(gamma_test):<10.1f} {result_str:<15} {status:<8}")
            
            # Stop at first infinity to avoid spam
            if not np.isfinite(result):
                print(f"   *** First infinity found at gamma = {np.degrees(gamma_test):.1f}° ***")
                break
                
        except Exception as e:
            print(f"   {np.degrees(gamma_test):<10.1f} ERROR: {e}")
            break

# Call the debugging function
debug_parameter_ranges(source_positions, detector_positions, center_mm)

# %%

# %%
# Analyze why the integral becomes small/unstable for phi ≈ 0 and large gamma
def analyze_phi_zero_gamma_behavior():
    """Analyze the mathematical behavior when phi ≈ 0 and gamma is large"""
    print("\nANALYZING PHI ≈ 0, LARGE GAMMA BEHAVIOR")
    print("="*60)
    
    # Use parameters from our test case
    test_source = source_positions[0]
    R_src, source_angle = get_source_params(test_source.reshape(1, -1), center_mm)
    R_src = R_src[0]
    source_angle = source_angle[0]
    
    print(f"Using source: R={R_src:.2f}mm, angle={np.degrees(source_angle):.1f}°")
    print(f"Brain radius: {R_brain}mm, Skull thickness: {h_skull}mm")
    
    # Test specific phi ≈ 0, varying gamma
    phi_test = 0.0  # Exactly zero
    gamma_values = np.linspace(0, np.pi/3, 10)  # 0° to 60°
    
    print(f"\nAnalyzing phi = {np.degrees(phi_test):.1f}°, varying gamma:")
    print(f"{'Gamma (°)':<10} {'Distance d':<12} {'Term2':<12} {'Expo':<12} {'Result':<12}")
    print("-" * 65)
    
    from guti.modalities.us.pat_analytical_integrated import analytical_integrand
    import jax.numpy as jnp
    
    for gamma in gamma_values:
        # Manually compute the problematic terms to understand the issue
        
        # Parameters from analytical_integrand
        r = R_brain
        alpha = alpha_brain  
        beta = beta_skull
        h = h_skull
        t = np.tan(source_angle)
        
        # Distance calculation - this is where the problem starts
        d_squared = (
            (r * np.sin(gamma) - R_src * t)**2
            + R_src**2  
            + (r * np.cos(gamma))**2
            - 2 * r * R_src * np.cos(phi_test)  # cos(0) = 1
        )
        
        d = np.sqrt(max(d_squared, 1e-12))  # Prevent negative sqrt
        
        # When phi = 0, sin(phi) = 0, so some terms simplify
        # Using Dphi=0, Dgamma=0 for the center of integration
        Dphi = 0.0
        Dgamma = 0.0
        
        term1 = h**2 + (r * Dphi)**2 + (r * Dgamma)**2  # = h^2
        term2 = (
            d
            + Dgamma * r * R_src * t * np.cos(gamma) / d  # = d (since Dgamma=0)
            - r * R_src * np.sin(phi_test) * Dphi / d     # = 0 (since sin(0)=0)
        )
        # So term2 = d
        
        expo_part1 = -beta * term1 * term2 / (d**2)  # = -beta * h^2 * d / d^2 = -beta * h^2 / d
        expo_part2 = alpha * (d 
                             - r * R_src * t * np.cos(gamma) / d
                             + r * R_src * np.sin(phi_test) / d)  # sin(0)=0
        # So expo_part2 = alpha * (d - r * R_src * t * cos(gamma) / d)
        
        expo = expo_part1 + expo_part2
        result = np.exp(expo)
        
        print(f"{np.degrees(gamma):<10.1f} {d:<12.3f} {term2:<12.3f} {expo:<12.3f} {result:<12.3e}")
        
    print(f"\nKey insights:")
    print(f"1. When phi = 0°, cos(phi) = 1, maximizing the -2*r*R*cos(phi) term")
    print(f"2. This makes the distance d smaller for certain gamma values")
    print(f"3. Small d leads to large negative exponential arguments → very small results")
    print(f"4. At extreme values, d can become so small that divisions by d explode")
    
    # Show the critical distance calculation in detail
    print(f"\nDistance calculation analysis (phi = 0°):")
    print(f"d² = (r*sin(γ) - R*tan(src_angle))² + R² + (r*cos(γ))² - 2*r*R*cos(0)")
    print(f"d² = (r*sin(γ) - R*tan(src_angle))² + R² + (r*cos(γ))² - 2*r*R")
    print(f"d² = ({R_brain:.1f}*sin(γ) - {R_src:.1f}*{np.tan(source_angle):.3f})² + {R_src:.1f}² + ({R_brain:.1f}*cos(γ))² - 2*{R_brain:.1f}*{R_src:.1f}")
    print(f"d² = ({R_brain:.1f}*sin(γ) - {R_src * np.tan(source_angle):.1f})² + {R_src**2:.1f} + ({R_brain:.1f}*cos(γ))² - {2*R_brain*R_src:.1f}")
    
    # Find the gamma where d becomes minimum
    gamma_fine = np.linspace(0, np.pi/2, 100)
    d_values = []
    
    for gamma in gamma_fine:
        d_squared = (
            (R_brain * np.sin(gamma) - R_src * np.tan(source_angle))**2
            + R_src**2
            + (R_brain * np.cos(gamma))**2  
            - 2 * R_brain * R_src
        )
        d_values.append(np.sqrt(max(d_squared, 0)))
    
    d_values = np.array(d_values)
    min_d_idx = np.argmin(d_values)
    min_d_gamma = gamma_fine[min_d_idx]
    min_d_value = d_values[min_d_idx]
    
    print(f"\nMinimum distance occurs at:")
    print(f"γ = {np.degrees(min_d_gamma):.1f}°, d_min = {min_d_value:.3f} mm")
    print(f"This explains the minimum in the integral around this gamma value!")
    
    # Plot the distance vs gamma
    plt.figure(figsize=(10, 6))
    plt.plot(np.degrees(gamma_fine), d_values, 'b-', linewidth=2)
    plt.axvline(np.degrees(min_d_gamma), color='red', linestyle='--', 
                label=f'Minimum at γ = {np.degrees(min_d_gamma):.1f}°')
    plt.xlabel('Gamma (degrees)')
    plt.ylabel('Distance d (mm)')
    plt.title('Distance d vs Gamma (at phi = 0°)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print(f"\nWhy this causes problems:")
    print(f"• Small d → Large terms like (1/d) in the exponential")
    print(f"• Large negative exponential arguments → Very small results")
    print(f"• At d → 0, terms become infinite → Numerical breakdown")
    print(f"• Your controlled plot avoided the most extreme cases")

# Run the analysis
analyze_phi_zero_gamma_behavior()

# %%
