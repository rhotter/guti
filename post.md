
A Unified Theory of Imaging
There are many ways to measure brain activity: EEG, MEG, fMRI, fNIRS, ultrasound, and more. How would you go about comparing them, quantitatively?

People usually start with resolution, analogous to the number of pixels in a camera sensor. But if you know about cameras, you know that the number of megapixels in your camera is far from a complete picture of how good your camera is. 

The especially weird thing about resolution for brain imaging is that a global resolution is not even well defined, which is why so many papers disagree on the resolution of the same modality. The resolution depends on the depth you’re considering, the assumptions you make about the signal, like whether it’s sparse or not, and what algorithms you use to reconstruct your brain images.

What we really care about is information: how many bits per second can this device tell you about what’s going on in the brain? In this post, we’ll compute, from first principles, the theoretical information limit of all the popular brain imaging modalities.

The spoiler is we get a table like this:



We’ll explain how we get this and show scaling plots showing how the information scales with the number of sensors.
Brain imaging as a communication channel
Each imaging modality can be viewed as a communication channel between the brain and the sensors. The device transforms the state of the brain — for example, which neurons are firing and when — into measurements such as voltages on EEG electrodes, magnetic fields inwhich, for example, could be which neurons are firing and when — into measurements like voltages on EEG electrodes, magnetic fields for MEG, etc.

The human brain has about 100 billion neurons. So let’s represent the state of the brain as a 100-billion-dimensional vector x, where each element encodes whether a neuron is firing100 billion-dimensional vector, x, where each element encodes whetherif the neuron is firing or not. 

Then we can write the transformation from brain state to measurements, y, as some function

y = f(x(t))y=f(x(t))

In practice, our sensors aren’t perfect, so we also add noise.

y = f(x(t)) + ny=f(x(t)) + n(t)



You can think of the vector yy as, for example, voltage measurements for each electrode in EEG. f is a physical simulator of the modality; it tells you, in the absence of noise, what measurements you’d get for a given brain state. We call f the forward model.

> How do we get the forward model f?
We’ll explain that later, but the rough idea is that there are equations from physics that you can use to model each modality. Then you can write a simulator that solves those equations.

Let’s go back to our original goal of measuring information. What we’re trying to do is study the function f, and ask how much of the brain state is actually represented in the measurement, and how much gets lost.

For most imaging modalities, the variations in x are small enough that the function is close to linear, so we can represent the function f as a matrix, A.

y = Ax + n

So now it becomes a linear algebra problem! The linear algebra question we’re asking is something like “how invertible is the matrix?” If it’s invertible, then you can perfectly recover the brain state from the measurements.
Matrix inversion in the real world
Usually in linear algebra, you think of matrix inversion as a binary thing — either the matrix is invertible or it’s not. But it’s actually more of a continuous thing once you move to the real world, where you have noise in your measurements.

The less binary way of asking how invertible a matrix is, is by looking at its singular values.

The singular value decomposition (SVD) gives you a way to decompose the brain state x into components (called singular vectors), and the singular value tells you how much each component gets amplified or attenuated in your measurement. Think of each singular value as a gain in an amplifier circuit. For example, if the i’th singular value is 0, that means all the information in x related to the i’th singular vector gets lost. But it’s not binary. If the singular value is very small, that part of the measurement will also get drowned out in noise.

Let’s take an example of blurring a signal. Here’s the singular value spectrum of a Gaussian blur. In these spectrum plots, we sort the singular values from largest to smallest.

[Gaussian Blur SVD]


You see that initially the spectrum is high, but then it drops off exponentially. So when you blur a signal x, there are components of that signal that get very strongly attenuated. Let’s look at what those components look like!



The early singular vectors are low frequency, the high ones oscillate a lot. So blurs attenuate pieces that oscillate a lot. This makes sense: if you blur an image with sharp edges (i.e. the neighboring pixels change drastically), the sharp edges go away.

> Fun math fact
An interesting fact is that the singular vectors of a blur are precisely the fourier basis vectors and the singular values are equal to the fourier transform of the signal.
Can you undo a blur?
If you have no noise, the answer is actually yes! With perfect knowledge of the forward model (in our case, the precise degree of blurring represented by the width of the Gaussian), since the singular values don’t go to 0, we can invert the forward model, and get back the unblurred signal. So in the noiseless case, you haven’t lost any information!

The problem is when you add noise. The picture is like this:



In the early components, the signal lies above the noise. But after some singular value, the singular value components get drowned out by noise. So after blurring, you have forever lost the small singular value pieces of your signal.

> Why is the noise a flat line?
We assume the noise is independent and identically distributed across detectors. In this case, the variance of the values <n, u_i> will be the same for all singular vectors u_i.

> But can’t machine learning undo a blur? Isn’t super-resolution a thing in machine learning?
Yes! There’s a field of machine learning known as super-resolution or upsampling. It seems like you can magically increase the resolution of images.



But the key to super resolution in machine learning is that you assume a prior over the possible signal vectors x. For example, you know that natural images are more likely than random noise, so you can guess the later singular components from the earlier ones.

But if your signal was truly random gaussian noise, no machine learning method will recover it! The early components of x will not tell you anything about the late components of x.

Could machine learning super resolution be applied to brain imaging? Totally. And this is another reason we don’t like the resolution metric. But the key is that machine learning cannot increase the channel capacity of the system.

SVDs of brain imaging modalities
We turned a bunch of brain imaging modalities into matrices, and computed the singular value decomposition of all of them.

Here’s the singular value spectrum of all of them, normalized by their first singular value.


We can see that modalities like EEG and MEG fall off much more quickly than fNIRS and ultrasound. Critically though, this is for a single time sample, and doesn’t include the temporal dynamics where EEG and MEG shine (more on that later).

> How many sensors do you use for each modality?
These spectrums actually represent the case where you have infinite sensors! Of course, you can’t simulate infinite sensors on a computer, but we ran scaling experiments which showed convergence in the spectrum.

> How do you simulate a brain imaging modality? What does that even mean?
Each modality is governed by partial differential equations which describe its physics.

For example, in fNIRS, you send near-infrared light through the brain. If you know how much each point in the head scatters and absorbs light, you can predict where the light will go. The equation which describes this is called the Radiative Transfer Equation, and is actually also used to model nuclear reactors!

Once you have equations for the physics, you can compute the derivative of the sensor reading with respect to the brain properties (either analytically or with a differentiable partial differential equation solver). This derivative gives you the matrix A for each modality.

All our experiments were done on a simple model of the head. We model the head as a hemisphere with 3 layers — scalp (thickness of X cm), skull (thickness of Y cm), and brain (radius of Z cm).

Here’s a table of the equations used for each modality:


Modality
Differential equation
Package used
Head properties assumed
EEG

OpenMEEG
Conductivity of brain = conductivity of scalp = 33x conductivity of skull
MEG






fNIRS
Radiative transfer equation
[write math]
Analytical expression for half plane Green’s function





EEG


Package used: OpenMEEG
MEG


fNIRS


Ultrasound






There was a ton of details and engineering needed to compute these SVDs. A single simulation could take hours to run, even on a GPU, and the matrices A had up to 100 billion elements. This is on the same scale as the number of parameters in a large language model! To run SVD on such large matrices required writing our own SVD implementation that would run on multiple GPUs.






We can count the number of components that are above the noise floor. That tells us roughly how many independent components of the brain state get preserved in our measurements. But how you define that crossover point is kind of arbitrary: instead of the crossover being at where the signal equals the noise, we could have alternatively defined it to be where the signal was 10x above the noise. There’s a more rigorous way to think about this: information.







The noise
To compare modalities, we need an explicit sensor-noise model for each one.

At the code level, we now separate two effects:

1. A physical detector floor, such as electrode Johnson noise in EEG, field-noise density in MEG, detector shot noise in fNIRS, and minimum-detectable pressure in ultrasound.
2. A sensor-count scaling rule. If total head coverage is fixed and we add more sensors, some modalities must shrink the effective sensor area, which raises the noise floor.

The scaling assumptions we currently use are:

- EEG: noise grows like sqrt(N) at fixed scalp coverage, since Johnson voltage noise scales like sqrt(R) and contact resistance rises as electrode area shrinks.
- MEG OPM: a conservative sqrt(N) penalty under fixed helmet coverage.
- MEG SQUID: an N penalty, because pickup-loop field noise scales roughly like 1 / area if flux noise is approximately fixed.
- CW fNIRS and TD fNIRS: sqrt(N), matching shot-noise-limited detectors with shrinking aperture area.
- Ultrasound: for now a conservative sqrt(N) penalty, until the receive model is calibrated in absolute detector units.

For the reference detector floors, we use numbers in the range of current hardware: about 0.4 uV RMS for EEG front ends over roughly 0.5-100 Hz, about 3 fT/sqrt(Hz) for SQUID magnetometers, about 15 fT/sqrt(Hz) for OPMs, about 17.7 fW/sqrt(Hz) for a high-performance fNIRS detector, and sub-Pa/sqrt(Hz) minimum-detectable pressure for modern optical ultrasound detectors.

Because not every forward model in this repo is yet calibrated all the way to absolute physical units, the capacity code carries one extra per-modality calibration constant: an effective reference output SNR at a reference sensor count. The key change is that the noise floor is no longer inferred from the singular spectrum itself, and the sensor-count dependence is now explicit.

Channel capacity
Let’s move from linear algebra back to information theory. We have a communication channel between the brain and the sensors.

Brain —> Sensors

Every communication channel has a limit to how much information can be sent through it. It’s called the channel capacity. If the noise is Gaussian and uncorrelated in time, you can actually write a precise mathematical expression for the channel capacity. For each sample, it’s

I = ½ log(1 + SNR)









































We can use the channel capacity as a metric to estimate the amount of information that can be maximally retrieved from the brain. We can compute it directly from the SVD spectrum: we have N independent dimensions which are above the noise floor. For each of them we know the amount of signal and the level of noise, and can compute the signal-to-noise ratio. The bitrate for one component is Ci=12 f  log (1+PiN), where Pi is the power sent over that component, and f is the sampling frequency. In reality, you have a fixed “power budget” and you distribute it across components. For the max bitrate, we optimize the power distribution to get the most information from the sensors.

Temporal dynamics
In the Gaussian blur example, as well as in most of the computations, we’ve assumed a stationary system and one recording from it. What you actually get in reality is a recording on each sensor over time, though. If you assume each time point is independent from each other, the SVD spectrum of the resulting measurement operator consists of the same values, repeated for each time point. 
