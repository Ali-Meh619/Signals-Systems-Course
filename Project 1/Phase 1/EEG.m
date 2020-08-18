%EEG
%Plotting frequency domain
load Subject1.mat;
load filter_1.mat

m1=SubjectData(1,:);
m2=SubjectData(2,:);
T=m1(1,2)-m1(1,1);
fs=1/T;
L=length(m1);

Y = fft(m2);
Y(1)=0;
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;

plot(f,P1) ;
title('Single-Sided Amplitude Spectrum of X(t)');
xlabel('f (Hz)');
ylabel('|P1(f)|');
%%
%cutoff frequency
j=0.5;
o=0;
while(o<0.99)
   if  (bandpower(mk,fs,[0 j])/ bandpower(mk,fs,[0 40]))<0.99
         
      j=j+0.5;
 o=(bandpower(mk,fs,[0 j])/ bandpower(mk,fs,[0 40]));
   end
end
%%
%stimulioneset
u=zeros(9,L);

u(1,:)=SubjectData(1,:);
z=0;
h=zeros(1,2700);
for i=1:L
    if SubjectData(10,i) ~= 0 
        
        if SubjectData(10,i-1)==0

       
                  z=z+1;
                h(1,z) = i;
          
        end
    end
end
%%
%filtering Subjecdata
   for i=2:9  
   u(i,:)=filter(filter_1,SubjectData(i,:));
   end
   %%
   %plotting filtered signal
X = fft(u(5,:,1));


P22 = abs(X/L);
P11 = P22(1:L/2+1);
P11(2:end-1) = 2*P11(2:end-1);

    figure;
     subplot(2,1,1);
    plot(f,P1);
    title('Power spectrum of the original signal');
    

    subplot(2,1,2);
    plot(f,P11);
    title('filtered signal');
   %%
   %filtered signal in time domain
   time = 1/fs*(0:L-1)';
       figure;
    subplot(2,1,1);
    plot(time,SubjectData(5,:,1));
    title('The original time series');
    subplot(2,1,2);
    plot(time,u(5,:,1));
    title('The band-pass filtered time series');
   
   %%
   %epoching
   Epoch=epoching(u,0.2,0.8,h);
   
   
   
   %%
   %Calculating Energy BANDS
   b=[0 4;4 7;8 12;12 30;30 40];
   w=size(b,1);
E=zeros(w,2700);
for oo=1:w
   for ll=1:2700
       for tt=2:9
   
   E(oo,ll)=a(oo,ll)+freqband(Epoch(tt,:,ll),b(oo,1),b(oo,2),256);
   
       end
   end
end





%%

function y=epoching (o,Back_s,Forward_s,h)

l=length(o);
p=length(h);
m1=o(1,:);
m2=o(2,:);
T=m1(1,2)-m1(1,1);
sb=floor(Back_s/T);
sf=floor(Forward_s/T);
EPOCH=zeros(9,sb+sf+1,p);
for i=1:p                
    EPOCH(:,:,i) = o(1:9,h(1,i)-sb:h(1,i)+sf);   
     end
y=EPOCH;
end

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