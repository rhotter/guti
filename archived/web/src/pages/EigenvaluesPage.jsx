import { math } from 'mathjs';
import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { KatexBlock } from '../components/KatexBlock';

const EigenvaluesPage = () => {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    // Generate data for visualization
    const generateData = () => {
      // Size of our matrix
      const N = 30;
      
      // Create a symmetric matrix (for real eigenvalues)
      // We'll use a simple tridiagonal matrix
      const matrix = [];
      for (let i = 0; i < N; i++) {
        const row = Array(N).fill(0);
        row[i] = 2; // Diagonal elements
        if (i > 0) row[i-1] = -1; // Lower diagonal
        if (i < N-1) row[i+1] = -1; // Upper diagonal
        matrix.push(row);
      }

      // Calculate eigenvalues and eigenvectors
      // For simplicity, we're using analytical values for this specific tridiagonal matrix
      // In practice, we'd use a numerical method
      const eigenvalues = [];
      const eigenvectors = [];
      
      for (let k = 1; k <= N; k++) {
        // For this specific tridiagonal matrix, eigenvalues are:
        const lambda = 2 - 2 * Math.cos(k * Math.PI / (N + 1));
        eigenvalues.push(lambda);
        
        // And eigenvectors are:
        const v = [];
        for (let i = 1; i <= N; i++) {
          v.push(Math.sin(i * k * Math.PI / (N + 1)));
        }
        // Normalize the eigenvector
        const norm = Math.sqrt(v.reduce((sum, val) => sum + val*val, 0));
        eigenvectors.push(v.map(val => val / norm));
      }

      // For visualization, we'll demonstrate how an eigenvector remains in the same direction
      // when multiplied by the matrix, just scaled by the eigenvalue
      const selectedIndex = 2; // Choose the third eigenvector for demonstration
      const selectedEigenvector = eigenvectors[selectedIndex];
      const selectedEigenvalue = eigenvalues[selectedIndex];
      
      // Calculate Av for the selected eigenvector
      const matrixTimesEigenvector = math.multiply(matrix, selectedEigenvector);

      return {
        matrix,
        eigenvalues,
        eigenvectors,
        selectedEigenvector,
        selectedEigenvalue,
        matrixTimesEigenvector
      };
    };

    setData(generateData());
  }, []);

  if (!data) {
    return <div>Loading visualization...</div>;
  }

  return (
    <div className="page-container">
      <h1>Eigenvalues and Eigenvectors Visualization</h1>
      
      <section className="explanation">
        <h2>Eigenvalues and Linear Transformations</h2>
        <p>
          Eigenvalues and eigenvectors provide insight into the behavior of linear transformations.
          An eigenvector of a linear transformation is a non-zero vector that changes only by a scalar factor when that linear transformation is applied to it.
        </p>
        
        <KatexBlock math="A\vec{v} = \lambda\vec{v}" />
        
        <p>
          Where A is a matrix, v is an eigenvector, and λ is the corresponding eigenvalue.
          Fourier basis vectors are eigenvectors of circulant matrices, which represent convolution operations.
        </p>
      </section>

      <section className="visualization">
        <h2>Visualization</h2>
        <div className="plot-container">
          <Plot
            data={[
              {
                z: data.matrix,
                type: 'heatmap',
                colorscale: 'Viridis'
              }
            ]}
            layout={{
              width: 600,
              height: 500,
              title: 'Matrix A (Tridiagonal)',
              xaxis: { title: 'Column Index' },
              yaxis: { title: 'Row Index' }
            }}
          />
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                x: Array.from({ length: data.eigenvalues.length }, (_, i) => i+1),
                y: data.eigenvalues,
                type: 'scatter',
                mode: 'markers',
                name: 'Eigenvalues'
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Eigenvalues',
              xaxis: { title: 'Index' },
              yaxis: { title: 'Value' }
            }}
          />
        </div>

        <div className="plot-container">
          <h3>Eigenvector Demonstration</h3>
          <p>
            Selected eigenvalue (λ₃): {data.selectedEigenvalue.toFixed(4)}
          </p>
          <Plot
            data={[
              {
                x: Array.from({ length: data.selectedEigenvector.length }, (_, i) => i+1),
                y: data.selectedEigenvector,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Eigenvector v₃'
              },
              {
                x: Array.from({ length: data.matrixTimesEigenvector.length }, (_, i) => i+1),
                y: data.matrixTimesEigenvector,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'A·v₃',
                line: { dash: 'dash' }
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Eigenvector v₃ and A·v₃',
              xaxis: { title: 'Component Index' },
              yaxis: { title: 'Value' },
              annotations: [
                {
                  x: 15,
                  y: 0.3,
                  text: 'A·v = λ·v',
                  showarrow: false,
                  font: { size: 16 }
                }
              ]
            }}
          />
        </div>
      </section>

      <section className="future-work">
        <h3>Interactive Elements to be Added:</h3>
        <ul>
          <li>Interactive matrix editor</li>
          <li>Eigenvector visualization in 2D/3D space</li>
          <li>Demonstration of how Fourier basis vectors are eigenvectors of convolution operations</li>
          <li>Connection between eigenvalues and frequency response</li>
        </ul>
      </section>
    </div>
  );
};

export default EigenvaluesPage; 