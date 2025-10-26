// Import required styles
import 'katex/dist/katex.min.css';
import './App.css';
import './index.css';

// Import math.js and plotly for our visualizations
import * as math from 'mathjs';
import Plotly from 'plotly.js';

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
  // Back to Top button logic
  const backToTopButton = document.getElementById('back-to-top');
  if (backToTopButton) {
    backToTopButton.addEventListener('click', () => {
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });
  }

  // Show/hide back to top button on scroll
  window.addEventListener('scroll', () => {
    if (backToTopButton) {
      if (window.scrollY > 300) {
        backToTopButton.style.display = 'flex';
      } else {
        backToTopButton.style.display = 'none';
      }
    }
  });

  // Handle active navigation highlighting
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.main-nav a');

  window.addEventListener('scroll', () => {
    let scrollY = window.scrollY;
    
    sections.forEach(section => {
      const sectionHeight = section.offsetHeight;
      const sectionTop = section.offsetTop - 100;
      const sectionId = section.getAttribute('id');
      
      if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
        navLinks.forEach(link => {
          link.classList.remove('active');
          if (link.getAttribute('href') === `#${sectionId}`) {
            link.classList.add('active');
          }
        });
      }
    });
  });

  // Function to create KaTeX formula
  function renderMath(container, formula, isBlock = true) {
    if (!container) return;
    
    // Directly render with KaTeX
    katex.render(formula, container, {
      displayMode: isBlock,
      throwOnError: false
    });
  }

  // Initialize all visualizations after a small delay to ensure DOM is fully processed
  setTimeout(() => {
    initLinearModelVisualization();
    initFourierVisualization();
    initEigenvaluesVisualization();
    initOrthogonalityVisualization();
    initChannelCodingVisualization();
  }, 100);
});

// Linear Model Visualization
function initLinearModelVisualization() {
  const container = document.getElementById('linear-model-viz');
  if (!container) return;

  // Generate data
  const vectorSize = 50;
  const originalVector = Array(vectorSize).fill(0).map((_, i) => 
    i > 15 && i < 35 ? 1 : 0 + 0.2 * Math.random() - 0.1);
  
  // Generate a simple smoothing matrix (Gaussian kernel)
  const smoothingMatrix = [];
  for (let i = 0; i < vectorSize; i++) {
    const row = [];
    for (let j = 0; j < vectorSize; j++) {
      // Gaussian kernel
      row.push(Math.exp(-0.1 * Math.pow(i - j, 2)));
    }
    smoothingMatrix.push(row);
  }

  // Normalize rows
  for (let i = 0; i < vectorSize; i++) {
    const sum = smoothingMatrix[i].reduce((a, b) => a + b, 0);
    for (let j = 0; j < vectorSize; j++) {
      smoothingMatrix[i][j] /= sum;
    }
  }

  // Apply the smoothing matrix
  const smoothedVector = math.multiply(smoothingMatrix, originalVector);
  
  // Render vector plot
  const vectorPlot = document.createElement('div');
  vectorPlot.className = 'plot-container';
  container.appendChild(vectorPlot);
  
  Plotly.newPlot(vectorPlot, [
    {
      x: Array.from({ length: originalVector.length }, (_, i) => i),
      y: originalVector,
      type: 'scatter',
      mode: 'lines',
      name: 'Original Vector'
    },
    {
      x: Array.from({ length: smoothedVector.length }, (_, i) => i),
      y: Array.isArray(smoothedVector) ? smoothedVector : smoothedVector.toArray(),
      type: 'scatter',
      mode: 'lines',
      name: 'Smoothed Vector'
    }
  ], {
    width: 800,
    height: 400,
    title: 'Original vs Smoothed Vector',
    xaxis: { title: 'Position' },
    yaxis: { title: 'Value' }
  });
  
  // Render matrix heatmap
  const matrixPlot = document.createElement('div');
  matrixPlot.className = 'plot-container';
  container.appendChild(matrixPlot);
  
  Plotly.newPlot(matrixPlot, [
    {
      z: smoothingMatrix,
      type: 'heatmap',
      colorscale: 'Viridis'
    }
  ], {
    width: 600,
    height: 500,
    title: 'Smoothing Matrix (Gaussian Kernel)',
    xaxis: { title: 'Column Index' },
    yaxis: { title: 'Row Index' }
  });

  // Render formula
  const mathContainer = document.getElementById('linear-model-formula');
  if (mathContainer) {
    renderMath(mathContainer, "y = Ax");
  }
}

// Fourier Visualization
function initFourierVisualization() {
  const container = document.getElementById('fourier-viz');
  if (!container) return;

  // Generate a signal with multiple frequency components
  const N = 1000;
  const t = Array.from({ length: N }, (_, i) => i / N * 10); // Time from 0 to 10
  
  // Signal with multiple frequency components
  const signal = t.map(time => 
    3 * Math.sin(2 * Math.PI * 0.5 * time) + // 0.5 Hz component
    2 * Math.sin(2 * Math.PI * 1.5 * time) + // 1.5 Hz component
    Math.sin(2 * Math.PI * 3 * time) +       // 3 Hz component
    0.5 * Math.random()                      // Noise
  );
  
  // Compute the Fourier transform
  // For this visualization, we use FFT from mathjs
  // In a real app, you might use a more optimized FFT library
  const fft = math.fft(signal);
  
  // Calculate magnitude spectrum
  const magnitudes = fft.map(c => Math.sqrt(c.re * c.re + c.im * c.im));
  const frequencies = Array.from({ length: N }, (_, i) => i * 10 / N);
  
  // Create signal plot
  const signalPlot = document.createElement('div');
  signalPlot.className = 'plot-container';
  container.appendChild(signalPlot);
  
  Plotly.newPlot(signalPlot, [
    {
      x: t,
      y: signal,
      type: 'scatter',
      mode: 'lines',
      name: 'Time Domain Signal'
    }
  ], {
    width: 800,
    height: 300,
    title: 'Time Domain Signal',
    xaxis: { title: 'Time (s)' },
    yaxis: { title: 'Amplitude' }
  });
  
  // Create frequency spectrum plot
  const fftPlot = document.createElement('div');
  fftPlot.className = 'plot-container';
  container.appendChild(fftPlot);
  
  // Only show the first half of the spectrum (up to Nyquist frequency)
  const nyquist = Math.floor(N/2);
  
  Plotly.newPlot(fftPlot, [
    {
      x: frequencies.slice(0, nyquist),
      y: magnitudes.slice(0, nyquist),
      type: 'scatter',
      mode: 'lines',
      name: 'Frequency Spectrum'
    }
  ], {
    width: 800,
    height: 300,
    title: 'Frequency Spectrum',
    xaxis: { title: 'Frequency (Hz)' },
    yaxis: { title: 'Magnitude' }
  });
  
  // Smoothing filter in frequency domain (low-pass)
  const cutoffFreq = 2.0; // Hz
  const filter = frequencies.map(f => (f < cutoffFreq) ? 1 : 0);
  
  // Apply filter in frequency domain
  const filteredFFT = fft.map((c, i) => {
    const f = i * 10 / N;
    return (f < cutoffFreq) ? c : { re: 0, im: 0 };
  });
  
  // Inverse FFT to get filtered signal
  const filteredSignal = math.ifft(filteredFFT).map(c => c.re);
  
  // Create filtered signal plot
  const filteredPlot = document.createElement('div');
  filteredPlot.className = 'plot-container';
  container.appendChild(filteredPlot);
  
  Plotly.newPlot(filteredPlot, [
    {
      x: t,
      y: signal,
      type: 'scatter',
      mode: 'lines',
      name: 'Original Signal',
      opacity: 0.5
    },
    {
      x: t,
      y: filteredSignal,
      type: 'scatter',
      mode: 'lines',
      name: 'Filtered Signal',
      line: { color: 'red' }
    }
  ], {
    width: 800,
    height: 300,
    title: 'Original vs Filtered Signal (Low-pass, Cutoff = 2Hz)',
    xaxis: { title: 'Time (s)' },
    yaxis: { title: 'Amplitude' }
  });
  
  // Render Fourier formula
  const mathContainer = document.getElementById('fourier-formula');
  if (mathContainer) {
    renderMath(mathContainer, "\\mathcal{F}\\{Ax\\} = \\mathcal{F}\\{A\\} \\cdot \\mathcal{F}\\{x\\}");
  }
}

// Eigenvalues Visualization
function initEigenvaluesVisualization() {
  const container = document.getElementById('eigenvalues-viz');
  if (!container) return;

  // Create a simple 2x2 matrix for demonstration
  const A = math.matrix([[4, 1], [1, 3]]);
  
  // Compute eigenvalues and eigenvectors
  const eig = math.eigs(A);
  const eigenvalues = eig.values;
  const eigenvectors = eig.vectors;
  
  // Create a grid of points to visualize the transformation
  const gridSize = 20;
  const grid = [];
  for (let i = -gridSize/2; i <= gridSize/2; i++) {
    for (let j = -gridSize/2; j <= gridSize/2; j++) {
      grid.push([i, j]);
    }
  }
  
  // Apply the transformation to each point
  const transformedGrid = grid.map(point => {
    const result = math.multiply(A, point);
    return Array.isArray(result) ? result : result.toArray();
  });
  
  // Create the eigenvector visualization
  const eigenvectorPlot = document.createElement('div');
  eigenvectorPlot.className = 'plot-container';
  container.appendChild(eigenvectorPlot);
  
  // Scale the eigenvectors for better visualization
  const scaleFactor = 5;
  const ev1 = math.multiply(scaleFactor, math.column(eigenvectors, 0)).toArray().flat();
  const ev2 = math.multiply(scaleFactor, math.column(eigenvectors, 1)).toArray().flat();
  
  // Create the eigenvector trace
  const ev1Transformed = math.multiply(A, math.column(eigenvectors, 0)).toArray().flat();
  const ev2Transformed = math.multiply(A, math.column(eigenvectors, 1)).toArray().flat();
  
  const traces = [
    // Original grid
    {
      x: grid.map(p => p[0]),
      y: grid.map(p => p[1]),
      mode: 'markers',
      type: 'scatter',
      marker: { size: 3, color: 'blue', opacity: 0.3 },
      name: 'Original Grid'
    },
    // Transformed grid
    {
      x: transformedGrid.map(p => p[0]),
      y: transformedGrid.map(p => p[1]),
      mode: 'markers',
      type: 'scatter',
      marker: { size: 3, color: 'red', opacity: 0.3 },
      name: 'Transformed Grid'
    },
    // First eigenvector
    {
      x: [0, ev1[0]],
      y: [0, ev1[1]],
      mode: 'lines+markers',
      type: 'scatter',
      line: { width: 3, color: 'green' },
      marker: { size: 8, color: 'green' },
      name: `Eigenvector 1 (λ = ${eigenvalues[0].toFixed(2)})`
    },
    // Second eigenvector
    {
      x: [0, ev2[0]],
      y: [0, ev2[1]],
      mode: 'lines+markers',
      type: 'scatter',
      line: { width: 3, color: 'purple' },
      marker: { size: 8, color: 'purple' },
      name: `Eigenvector 2 (λ = ${eigenvalues[1].toFixed(2)})`
    },
    // First eigenvector transformed
    {
      x: [0, ev1Transformed[0]],
      y: [0, ev1Transformed[1]],
      mode: 'lines+markers',
      type: 'scatter',
      line: { width: 3, color: 'green', dash: 'dash' },
      marker: { size: 8, color: 'green' },
      name: 'Transformed Eigenvector 1'
    },
    // Second eigenvector transformed
    {
      x: [0, ev2Transformed[0]],
      y: [0, ev2Transformed[1]],
      mode: 'lines+markers',
      type: 'scatter',
      line: { width: 3, color: 'purple', dash: 'dash' },
      marker: { size: 8, color: 'purple' },
      name: 'Transformed Eigenvector 2'
    }
  ];
  
  Plotly.newPlot(eigenvectorPlot, traces, {
    width: 700,
    height: 600,
    title: 'Matrix Transformation and Eigenvectors',
    xaxis: {
      title: 'x',
      range: [-15, 15],
      zeroline: true
    },
    yaxis: {
      title: 'y',
      range: [-15, 15],
      zeroline: true,
      scaleanchor: 'x',
      scaleratio: 1
    }
  });
  
  // Create matrix visualization
  const matrixDesc = document.createElement('div');
  matrixDesc.className = 'matrix-description';
  matrixDesc.innerHTML = `
    <h3>Matrix: A = [
      [${A.get([0, 0])}, ${A.get([0, 1])}],
      [${A.get([1, 0])}, ${A.get([1, 1])}]
    ]</h3>
    <p>Eigenvalues: λ₁ = ${eigenvalues[0].toFixed(2)}, λ₂ = ${eigenvalues[1].toFixed(2)}</p>
  `;
  container.appendChild(matrixDesc);
  
  // Render eigenvalue formula
  const mathContainer = document.getElementById('eigenvalues-formula');
  if (mathContainer) {
    renderMath(mathContainer, "A\\mathbf{v} = \\lambda \\mathbf{v}");
  }
}

// Orthogonality and Noise Visualization
function initOrthogonalityVisualization() {
  const container = document.getElementById('orthogonality-viz');
  if (!container) return;

  // Generate a clean signal
  const N = 256;
  const t = Array.from({ length: N }, (_, i) => i / N * 8); // Time from 0 to 8
  
  // Clean signal with a few frequency components
  const cleanSignal = t.map(time => 
    2 * Math.sin(2 * Math.PI * 0.5 * time) + 
    Math.sin(2 * Math.PI * 1.2 * time) + 
    0.5 * Math.sin(2 * Math.PI * 2.5 * time)
  );
  
  // Add noise to create a noisy signal
  const noiseLevel = 0.5;
  const noisySignal = cleanSignal.map(
    val => val + noiseLevel * (Math.random() * 2 - 1)
  );
  
  // Compute the FFT of both signals
  const cleanFFT = math.fft(cleanSignal);
  const noisyFFT = math.fft(noisySignal);
  
  // Calculate magnitude spectra
  const cleanMagnitudes = cleanFFT.map(c => Math.sqrt(c.re * c.re + c.im * c.im));
  const noisyMagnitudes = noisyFFT.map(c => Math.sqrt(c.re * c.re + c.im * c.im));
  const frequencies = Array.from({ length: N }, (_, i) => i * 8 / N);
  
  // Create time domain signal comparison
  const timePlot = document.createElement('div');
  timePlot.className = 'plot-container';
  container.appendChild(timePlot);
  
  Plotly.newPlot(timePlot, [
    {
      x: t,
      y: cleanSignal,
      type: 'scatter',
      mode: 'lines',
      name: 'Clean Signal'
    },
    {
      x: t,
      y: noisySignal,
      type: 'scatter',
      mode: 'lines',
      name: 'Noisy Signal',
      opacity: 0.7
    }
  ], {
    width: 800,
    height: 300,
    title: 'Time Domain: Clean vs Noisy Signal',
    xaxis: { title: 'Time (s)' },
    yaxis: { title: 'Amplitude' }
  });
  
  // Create frequency domain comparison
  const freqPlot = document.createElement('div');
  freqPlot.className = 'plot-container';
  container.appendChild(freqPlot);
  
  // Only show the first half of the spectrum (up to Nyquist frequency)
  const nyquist = Math.floor(N/2);
  
  Plotly.newPlot(freqPlot, [
    {
      x: frequencies.slice(0, nyquist),
      y: cleanMagnitudes.slice(0, nyquist),
      type: 'scatter',
      mode: 'lines',
      name: 'Clean Signal Spectrum'
    },
    {
      x: frequencies.slice(0, nyquist),
      y: noisyMagnitudes.slice(0, nyquist),
      type: 'scatter',
      mode: 'lines',
      name: 'Noisy Signal Spectrum',
      opacity: 0.7
    }
  ], {
    width: 800,
    height: 300,
    title: 'Frequency Domain: Clean vs Noisy Signal Spectrum',
    xaxis: { title: 'Frequency (Hz)' },
    yaxis: { title: 'Magnitude' }
  });
  
  // Calculate noise energy distribution
  const noisePower = Array(nyquist).fill(0);
  for (let i = 0; i < nyquist; i++) {
    const cleanPower = cleanMagnitudes[i] * cleanMagnitudes[i];
    const noisyPower = noisyMagnitudes[i] * noisyMagnitudes[i];
    noisePower[i] = Math.max(0, noisyPower - cleanPower); // Avoid negative values due to numerical precision
  }
  
  // Create noise distribution plot
  const noisePlot = document.createElement('div');
  noisePlot.className = 'plot-container';
  container.appendChild(noisePlot);
  
  Plotly.newPlot(noisePlot, [
    {
      x: frequencies.slice(0, nyquist),
      y: noisePower,
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      name: 'Noise Power Distribution'
    }
  ], {
    width: 800,
    height: 300,
    title: 'Noise Power Distribution in Frequency Domain',
    xaxis: { title: 'Frequency (Hz)' },
    yaxis: { title: 'Power' }
  });
  
  // Render orthogonality formula
  const mathContainer = document.getElementById('orthogonality-formula');
  if (mathContainer) {
    renderMath(mathContainer, "\\int_{-\\infty}^{\\infty} e^{i2\\pi f_1 t} e^{-i2\\pi f_2 t} dt = \\delta(f_1 - f_2)");
  }
}

// Channel Coding Visualization
function initChannelCodingVisualization() {
  const container = document.getElementById('channel-coding-viz');
  if (!container) return;

  // Parameters for water-filling demonstration
  const numChannels = 20;
  const totalPower = 100;
  
  // Create frequency-dependent noise profile (SNR = 1/noise)
  const noiseProfile = Array(numChannels).fill(0).map((_, i) => {
    // Create a noise profile that increases with frequency
    return 0.1 + 0.9 * (i / numChannels) * (i / numChannels);
  });
  
  // Water-filling power allocation
  const waterLevel = findWaterLevel(noiseProfile, totalPower);
  const powerAllocation = noiseProfile.map(noise => 
    Math.max(0, waterLevel - noise)
  );
  
  // Calculate resulting SNR for each channel
  const snrValues = noiseProfile.map((noise, i) => 
    powerAllocation[i] > 0 ? powerAllocation[i] / noise : 0
  );
  
  // Calculate channel capacity for each subchannel
  const capacities = snrValues.map(snr => 
    snr > 0 ? 0.5 * Math.log2(1 + snr) : 0
  );
  
  // Total capacity
  const totalCapacity = capacities.reduce((sum, c) => sum + c, 0);
  
  // Create channel profiles plot
  const channelPlot = document.createElement('div');
  channelPlot.className = 'plot-container';
  container.appendChild(channelPlot);
  
  const channelIndices = Array.from({ length: numChannels }, (_, i) => i);
  
  Plotly.newPlot(channelPlot, [
    {
      x: channelIndices,
      y: noiseProfile,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Noise Profile',
      line: { color: 'red' }
    }
  ], {
    width: 800,
    height: 300,
    title: 'Channel Noise Profile',
    xaxis: { title: 'Frequency Channel Index' },
    yaxis: { title: 'Noise Level' }
  });
  
  // Create water-filling visualization plot
  const waterfillingPlot = document.createElement('div');
  waterfillingPlot.className = 'plot-container';
  container.appendChild(waterfillingPlot);
  
  const waterLevelArray = Array(numChannels).fill(waterLevel);
  
  Plotly.newPlot(waterfillingPlot, [
    {
      x: channelIndices,
      y: noiseProfile,
      type: 'scatter',
      mode: 'lines',
      name: 'Noise Floor',
      line: { color: 'red' }
    },
    {
      x: channelIndices,
      y: waterLevelArray,
      type: 'scatter',
      mode: 'lines',
      name: 'Water Level',
      line: { dash: 'dash', color: 'blue' }
    },
    {
      x: channelIndices,
      y: noiseProfile.map((noise, i) => noise + powerAllocation[i]),
      type: 'scatter',
      mode: 'lines',
      name: 'Signal + Noise Level',
      fill: 'tonexty',
      fillcolor: 'rgba(0, 128, 255, 0.3)',
      line: { color: 'rgb(0, 128, 255)' }
    }
  ], {
    width: 800,
    height: 400,
    title: 'Water-Filling Power Allocation',
    xaxis: { title: 'Frequency Channel Index' },
    yaxis: { title: 'Level' }
  });
  
  // Create power allocation bar chart
  const powerPlot = document.createElement('div');
  powerPlot.className = 'plot-container';
  container.appendChild(powerPlot);
  
  Plotly.newPlot(powerPlot, [
    {
      x: channelIndices,
      y: powerAllocation,
      type: 'bar',
      name: 'Power Allocation',
      marker: { color: 'rgb(0, 128, 255)' }
    }
  ], {
    width: 800,
    height: 300,
    title: `Power Allocation (Total Power: ${totalPower.toFixed(2)})`,
    xaxis: { title: 'Frequency Channel Index' },
    yaxis: { title: 'Power' }
  });
  
  // Create capacity bar chart
  const capacityPlot = document.createElement('div');
  capacityPlot.className = 'plot-container';
  container.appendChild(capacityPlot);
  
  Plotly.newPlot(capacityPlot, [
    {
      x: channelIndices,
      y: capacities,
      type: 'bar',
      name: 'Channel Capacity',
      marker: { color: 'rgb(0, 200, 0)' }
    }
  ], {
    width: 800,
    height: 300,
    title: `Channel Capacity (Total: ${totalCapacity.toFixed(2)} bits/symbol)`,
    xaxis: { title: 'Frequency Channel Index' },
    yaxis: { title: 'Capacity (bits/symbol)' }
  });
  
  // Add capacity information
  const capacityInfo = document.createElement('div');
  capacityInfo.className = 'capacity-info';
  capacityInfo.innerHTML = `
    <h3>Total Channel Capacity: ${totalCapacity.toFixed(2)} bits/symbol</h3>
    <p>Using water-filling power allocation with total power = ${totalPower}</p>
  `;
  container.appendChild(capacityInfo);
  
  // Render channel coding formula
  const mathContainer = document.getElementById('channel-coding-formula');
  if (mathContainer) {
    renderMath(mathContainer, "C = \\sum_{i} \\frac{1}{2} \\log_2 \\left(1 + \\frac{P_i}{N_i}\\right)");
  }
}

// Helper function for water-filling algorithm
function findWaterLevel(noiseProfile, totalPower) {
  // Sort the noise values in ascending order for efficiency
  const sortedNoise = [...noiseProfile].sort((a, b) => a - b);
  const n = sortedNoise.length;
  
  // Initial estimate: distribute power equally
  let waterLevel = totalPower / n + sortedNoise[0];
  
  // Iteratively refine the water level
  for (let i = 0; i < 20; i++) {
    let allocatedPower = 0;
    
    // Calculate power allocation with current water level
    for (let j = 0; j < n; j++) {
      if (waterLevel > sortedNoise[j]) {
        allocatedPower += (waterLevel - sortedNoise[j]);
      }
    }
    
    // Adjust water level based on allocated power
    const scaleFactor = totalPower / allocatedPower;
    const adjustment = (scaleFactor - 1) * 0.5; // Damping factor for stability
    
    // Update water level
    waterLevel *= (1 + adjustment);
    
    // If we're close enough to the target power, break
    if (Math.abs(allocatedPower - totalPower) / totalPower < 0.001) {
      break;
    }
  }
  
  return waterLevel;
}
