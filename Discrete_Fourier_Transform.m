
% Question No 1: What did you expect to observe based on the convolution
% theorem?
% Define time and signals
t = 0:0.05:10;
f = sin(pi*t) .* sin(3*pi*t); % Modulated sinusoid
g = ones(1,length(t)); % Box function (moving average filter)
% Perform Convolution in Time Domain
fgc = conv(f, g, 'same'); 
% Compute Fourier Transforms
N = length(f) + length(g); % Define zero-padded FFT length
F = fft(f, N);
G = fft(g, N);
FG = F .* G; % Convolution Theorem (Multiplication in Frequency Domain)
fgc2 = ifft(FG); % Inverse FFT to get time-domain convolution
% Shifted FFT for Visualization
F_shift = fftshift(F);
G_shift = fftshift(G);
FG_shift = fftshift(FG);
% Frequency Axis
fs = 1 / (t(2) - t(1)); % Sampling frequency
f_axis = linspace(-fs/2, fs/2, N);
% Plot Time Domain Signals
figure;
subplot(3,1,1), plot(t, f), title('Original Signal f(t)'), xlabel('Time'), ylabel('Amplitude');
subplot(3,1,2), plot(t, g), title('Box Function g(t)'), xlabel('Time'), ylabel('Amplitude');
subplot(3,1,3), plot(t, fgc), title('Convolved Signal f * g (Time Domain)'), xlabel('Time'), ylabel('Amplitude');
% Plot Frequency Response
figure;
subplot(3,1,1), plot(f_axis, abs(F_shift)), title('Magnitude Spectrum of f(t)'), xlabel('Frequency'), ylabel('Magnitude');
subplot(3,1,2), plot(f_axis, abs(G_shift)), title('Magnitude Spectrum of g(t) (Sinc Function)'), xlabel('Frequency'), ylabel('Magnitude');
subplot(3,1,3), plot(f_axis, abs(FG_shift)), title('Magnitude Spectrum After Convolution (F .* G)'), xlabel('Frequency'), ylabel('Magnitude');
% Compute Error Between Methods
error_value = sum(abs(fgc(1:length(t)) - real(fgc2(1:length(t)))));
disp(['Error between time and frequency domain convolution: ', num2str(error_value)]);
% Question 2: Explain the steps in above process. Rotate square by 45
% degrees and calculate 2-D FFT. Identify and explain the effect of rotation on
% the spectrum and phase.
% Step 1: Create Binary Image (Half Black, Half White)
a = [zeros(256,128) ones(256,128)];
figure, imagesc(a), axis image, colormap gray, colorbar
% Compute and Visualize Fourier Transform
af = fftshift(fft2(a)); 
figure, imagesc(log(1+abs(af))), axis image, colormap gray, colorbar
figure, imagesc(angle(af)), axis image, colormap gray, colorbar
% Step 2: Create and Visualize a Square
[x,y] = meshgrid(1:256,1:256);
a = zeros(256,256);
a(78:178,78:178) = 1;
figure, imagesc(a), colormap gray, colorbar
% Compute and Visualize FFT of Square
af = fftshift(fft2(a));
figure, imagesc(log(1+abs(af))), axis image, colormap gray, colorbar
figure, imagesc(angle(af)), axis image, colormap gray, colorbar
% Step 3: Create and Visualize a Diamond Shape
b = (x+y<329) & (x+y>182) & (x-y>-67) & (x-y<73);
figure, imagesc(b), colormap gray, colorbar
% Compute and Visualize FFT of Diamond Shape
bf = fftshift(fft2(b));
figure, imagesc(log(1+abs(bf))), axis image, colormap gray, colorbar
figure, imagesc(angle(bf)), axis image, colormap gray, colorbar
% Step 4: Rotate the Diamond by 30 Degrees
c = imrotate(b, 30, 'nearest', 'crop');
figure, imagesc(c), axis image, colormap gray, colorbar
% Compute and Visualize FFT of Rotated Shape
cf = fftshift(fft2(c));
figure, imagesc(log(1+abs(cf))), axis image, colormap gray, colorbar
figure, imagesc(angle(cf)), axis image, colormap gray, colorbar
% Step 5: Rotate the Square by 45 Degrees and Analyze Effect
c_45 = imrotate(a, 45, 'nearest', 'crop');

figure, imagesc(c_45), axis image, colormap gray, colorbar
cf_45 = fftshift(fft2(c_45));
figure, imagesc(log(1+abs(cf_45))), axis image, colormap gray, colorbar
figure, imagesc(angle(cf_45)), axis image, colormap gray, colorbar
% Exercise 3
% Aliasing
A = imread('barbara.tif');
figure, imagesc(A), colormap gray, axis image, colorbar;
[row, col] = size(A);
% Downsampling by pixel replication.
X=1:4:row;
Y=1:4:col;
B = A(X,Y);
figure, imagesc(B), colormap gray, axis image, colorbar;
C = imresize(B, 4, 'nearest');
figure, imagesc(C), colormap gray, axis image, colorbar;
N = 9;
sigma = 1;
for x=1:N, for y=1:N,
h(x,y)=(1/(2*pi*sigma^2))*exp((-1)*((x-(N+1)/2)^2+(y-(N+1)/2)^2)/(2*sigma^2));,
end, end
% Perfrom 2D Correlation using con2 instead of Correlation2D
A2 = conv2(double(A), h, 'same');
B2 = A2(X,Y);
figure, imagesc(B2), colormap gray, axis image, colorbar;
C2 = imresize(B2, 4, 'nearest');
figure, imagesc(C2), colormap gray, axis image, colorbar;
% Downsampling by a factor of 4
X = 1:4:row;
Y = 1:4:col;
B = A(X,Y);
% Downsampling by a factor of 8
X = 1:8:row;
Y = 1:8:col;
B8 = A(X,Y);
figure, imagesc(B8), colormap gray, axis image, colorbar;
% Upsampling by a factor of 8
C8 = imresize(B8, 8, 'nearest');
figure, imagesc (C8), colormap gray, axis image, colorbar;
% Exercise 4: DFT
% Step 1: Read and Display the Test Image
A = imread('barbara.tif');
figure, imagesc(A), colormap gray, axis image, colorbar;
title('Original Image');
% Get Image Dimensions
[row, col] = size(A);
% Step 2: Apply 2-D FFT to Original Image
F = fft2(A);
% Step 3: Display Spectrum using Log Transform
figure, imagesc(log(1+abs(F))), colormap gray, axis image, colorbar;
title('Magnitude Spectrum (Before Shifting)');
% Step 4: Use fftshift to Shift Spectrum
F_shifted = fftshift(F);
% Step 5: Display Shifted Spectrum using Log Transform
figure, imagesc(log(1+abs(F_shifted))), colormap gray, axis image, colorbar;
title('Magnitude Spectrum (After fftshift)');
% Step 6: Multiply Original Image by (-1)^(x+y) to Shift in Spatial Domain
[X, Y] = meshgrid(1:col, 1:row);
A_shifted = double(A) .* (-1).^(X + Y);
% Step 7: Apply 2-D FFT to the Shifted Image
F2 = fft2(A_shifted);
% Step 8: Display Spectrum using Log Transform
figure, imagesc(log(1+abs(F2))), colormap gray, axis image, colorbar;
title('Magnitude Spectrum After Spatial Shift');
% Step 9: Display the Three Spectra as Surfaces
figure;
subplot(1,3,1), mesh(log(1+abs(F))), title('Original Spectrum');
subplot(1,3,2), mesh(log(1+abs(F_shifted))), title('Shifted Spectrum (fftshift)');

