%
function y=freqband(x,f1,f2,fs)
L=length(x);
dF = fs/L;
f  = (-fs/2:dF:fs/2-dF)';

if isempty(f1) || f1==-Inf
    BPF = (abs(f) < f2);
elseif isempty(f2) || f2==Inf
    BPF = (f1 < abs(f));
else
    BPF = ((f1 < abs(f)) & (abs(f) < f2));
end
Y=fftshift(fft(x));
P2 = abs(Y);
spec=P2.*BPF;
a=spec.*conj(spec);
y=(1/L)*(sum(a(:)));
end