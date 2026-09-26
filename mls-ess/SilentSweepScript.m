% Silence Sweep Signal synthesis
% Ref. A.Farina, unipr, Silence Sweep: a novel method for measuring electro-acoustical devices
% Industrial Engineering Dept., University of Parma, Via G.P. Usberti 181/A, 43100 Parma, ITALY
%                                    farina@unipr.it
% -------------------------
% ESS sequence 10*2^17 long
% -------------------------
% sampling frequency
wkdir='.\';
disp('ESS begin '); datetime
% Fs scelta
fs=48000;
% nbit=17;
nbit=17;
% lMLS length of MLS signal
lMLS=2^nbit-1;
% noct is P number of octaves
noct=10;
% lseq is L: the theoretical length of the ESS (floating point value)
lseq=noct*lMLS;
% nseq is N: the actual ESS length (equal to L rounded to integer)
nseq=round(lseq);
xESS=linspace(0, nseq-1, nseq);
noct22=2^noct;
yESS=sin((pi/(noct22)*lseq)/log(noct22).*exp(xESS/nseq*log(noct22)));
audiowrite(strcat(wkdir,'yESS.wav'),[yESS;yESS]',fs);
disp('ESS end');
% -------------------------------
% Time reversal of the ESS signal
% -------------------------------
disp('inv ESS begin ');datetime
yESSinv=zeros(1,nseq);
% k1=power(2,noct./nseq);
k2=noct*log(2)/(1-1/power(2,noct));
for i=1:nseq
	% yESSinv(i)=yESS(nseq-i+1)./power(k1,i).*k2;
    yESSinv(i)=yESS(nseq-i+1)/power(2,noct/nseq*i).*k2;
end
yESSinv=yESSinv./max(abs(yESSinv));
audiowrite(strcat(wkdir,'yESSinv.wav'),[yESSinv;yESSinv]',fs);
disp('inv ESS end');
% ----------------------------
% 41 MLS sequences 2^17-1 long
% ----------------------------
disp('MLS begin ');datetime
% nMLS=81;
nMLS=41;
yMLS=zeros(nMLS,lMLS);
for i=1:nMLS
    yMLS(i,1:lMLS)=mls(lMLS)';
end
% Central sequence zeroed
yMLS((nMLS-1)/2+1,:)=zeros(1,lMLS);
% Final MLS sequence
yMLSfin=zeros(1,nMLS.*lMLS);
for i=1:nMLS
    yMLSfin(1,(i-1)*lMLS+1:i*lMLS)=yMLS(i,1:lMLS);
end
audiowrite(strcat(wkdir,'yMLSfin.wav'),[yMLSfin;yMLSfin]',fs);
disp('MLS end');
% ------------------------------------------
% Convolution: MLS with ESS - very long time
% ------------------------------------------
disp('MLSESS convolution begin ');datetime
MLSESScnv=conv(yMLSfin,yESS);
% Normalize
MLSESScnv=MLSESScnv./max(abs(MLSESScnv));
% Full file written with tails 1310710 elements long on both sides
audiowrite(strcat(wkdir,'MLSESScnv.wav'),MLSESScnv,fs);
disp('MLSESS convolution end');
disp('MLSESS tail cut begin ');datetime
% Signal length with both tails
lMLSESScnv=length(MLSESScnv);
% Tail length
% ltail=nseq-noct;
ltail=nseq;
% Read refreshed MLSESScnv signal with no tails
[MLSESScnv, fs]=audioread(strcat(wkdir,'MLSESScnv.wav'),[ltail+1 lMLSESScnv-ltail]);
% Save mslESScnv file without tails.
audiowrite(strcat(wkdir,'MLSESScnv_notails.wav'),MLSESScnv,fs);
disp('MLSESS tail cut end   ');datetime