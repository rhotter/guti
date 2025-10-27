# Interactive Signal Processing Visualizations

This project provides interactive visualizations to help understand key concepts in signal processing, linear algebra, and information theory. It visually demonstrates how linear models, Fourier transforms, eigenvalues, orthogonality properties, and channel coding concepts relate to each other.

## Visualizations

1. **Linear Model Visualization**
   - Shows how a linear model can be visualized as a matrix of Gaussians multiplying a vector
   - Demonstrates the smoothing effect of Gaussian kernels on signals

2. **Fourier Transform Visualization**
   - Illustrates how linear operations in the time/space domain become multiplications in the Fourier domain
   - Shows frequency components of signals before and after filtering

3. **Eigenvalues Visualization**
   - Demonstrates how eigenvalues and eigenvectors relate to linear transformations
   - Shows how Fourier basis vectors are eigenvectors of convolution operations

4. **Orthogonality and Noise**
   - Visualizes the orthogonality property of Fourier transforms
   - Shows how noise gets converted to the same noise in Fourier space

5. **SNR and Channel Coding**
   - Applies Signal-to-Noise Ratio concepts to independent Fourier components
   - Demonstrates water-filling power allocation across frequencies

## Technologies Used

- React.js with Vite for fast development
- Plotly.js for interactive data visualizations
- Math.js for mathematical operations
- D3.js for custom visualizations
- KaTeX for mathematical equation rendering

## Getting Started

### Prerequisites

- Node.js (v14.x or higher)
- npm (v6.x or higher)

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd <repository-directory>/web
```

2. Install dependencies
```bash
npm install
```

3. Start the development server
```bash
npm run dev
```

4. Open your browser and navigate to `http://localhost:5173`

## Building for Production

To build the application for production:

```bash
npm run build
```

This will generate a `dist` folder with the production build.

To preview the production build locally:

```bash
npm run preview
```

## Project Structure

```
web/
├── src/
│   ├── components/       # Reusable components
│   ├── pages/            # Visualization pages
│   ├── utils/            # Utility functions
│   ├── App.jsx           # Main application component
│   ├── App.css           # Application styles
│   ├── main.jsx          # Entry point
│   └── index.css         # Global styles
├── public/               # Static assets
└── index.html            # HTML template
```

## Future Enhancements

- Add interactive controls for adjusting parameters in each visualization
- Implement animated transitions between different states
- Add more detailed explanations with mathematical derivations
- Create interactive tutorials with step-by-step guides
- Support for mobile devices with responsive layouts

## License

This project is licensed under the MIT License - see the LICENSE file for details.
