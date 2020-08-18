%
%% DFT 
clear; clc;
load y
signal = real(y);
F_trans = fft(signal);
norm1 = abs(F_trans);
phase1 = angle(F_trans);
n1 = 1:length(F_trans);
plot(n1,norm1)
xlabel('w')
ylabel('|X(w)|')
title('the norm of Fourier transform')
figure
plot(n1,phase1)
xlabel('w')
ylabel('phase(X(w))')
title('the phase of Fourier transform')
i = 2;
j = length(y);
norm_diff = 0;
phase_diff = 0;
for k = 1:length(y)
    i = i+1;
    j = j-1;
    if i >= j 
        break
    end
    norm_diff = norm_diff + abs(norm1(i)-norm1(j));
    phase_diff = phase_diff + abs(phase1(i)+phase1(j));
end
norm_diff;
phase_diff;
%% Sampling
clear; clc;
N = 90000;
x1 = ones(1,N);
for i = 0 : N-1
   if(i < (N/4))
       x1(i+1) = 1 - 4*i/N ;
   elseif ( i < 3*N/4 )
       x1(i+1) = 0;
   else
       x1(i+1) = 4*i/N - 3 ;
   end
end
x2 = ifft (x1);
x3 = [];
for i=1:N
   if mod(i,3) == 1
       x3 = [x3 x2(i)];
   end
end
ss=HalfBandFFT(x2,6);
x4 = abs(fft(x3));
Nprime = ceil(N/3);
k = (2*pi/Nprime)*[0:Nprime-1];
plot(k,x4)
axis([0 2*pi 0 0.4])