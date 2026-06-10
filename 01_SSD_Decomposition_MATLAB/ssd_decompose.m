function [trend, osc, noise, info] = ssd_decompose(inputData, L, outputPrefix)
% SSD/SSA decomposition for L-sensitivity experiment.
%
% inputData    : filename (CSV) or numeric vector
% L            : embedding length, e.g., 24 for hourly one-day window
% outputPrefix : optional prefix for saved files
%
% Example:
%   [trend,osc,noise,info] = ssd_decompose('speed.csv', 24);
%   [trend,osc,noise,info] = ssd_decompose(speed, 24, 'WSD1');

    % ---- Default arguments ----
    if ~exist('L','var') || isempty(L)
        L = 24;
    end

    if ~exist('outputPrefix','var') || isempty(outputPrefix)
        outputPrefix = sprintf('L%d', L);
    else
        outputPrefix = sprintf('%s_L%d', outputPrefix, L);
    end

    % ---- Load data ----
    if ischar(inputData) || isstring(inputData)
        T = readtable(inputData);

        if any(strcmpi(T.Properties.VariableNames, 'speed'))
            x = T{:, 'speed'};
        else
            x = T{:, 1};
        end
    else
        x = inputData;
    end

    x = double(x(:));
    N = numel(x);

    if N < 8
        error('Time series too short (N < 8).');
    end

    % ---- Validate L ----
    if ~(L > 2 && L < N/2)
        error('Invalid L. It must satisfy 2 < L < N/2. Current L = %d, N = %d.', L, N);
    end

    % ---- Fixed grouping parameters for L-sensitivity test ----
    low_freq_thresh = 0.05;   % trend if dominant frequency < 0.05 cycles/sample
    pair_tol        = 0.10;   % singular-value pairing tolerance, 10%
    osc_low         = 0.05;   % lower oscillation band
    osc_high        = 0.45;   % upper oscillation band

    % ---- Trajectory matrix ----
    K = N - L + 1;
    X = zeros(L, K);

    for i = 1:L
        X(i,:) = x(i:i+K-1).';
    end

    % ---- SVD ----
    [U,S,V] = svd(X,'econ');
    s = diag(S);
    r = numel(s);

    % ---- Reconstruct elementary components ----
    components = zeros(N, r);

    for i = 1:r
        Xi = s(i) * (U(:,i) * V(:,i).');
        components(:,i) = hankel_averaging(Xi);
    end

    % ---- Dominant frequency of each reconstructed component ----
    domFreq = zeros(r,1);

    for i = 1:r
        domFreq(i) = dom_freq(components(:,i));
    end

    % ---- Grouping: trend / oscillation / noise ----
    % Use < for trend to avoid overlap with oscillation at exactly 0.05.
    grp_trend = (domFreq < low_freq_thresh);
    grp_trend(1) = true;

    grp_osc = false(r,1);

    i = 2;
    while i <= r-1
        isPair = s(i) > 0 && ...
                 abs(s(i)-s(i+1))/max(s(i),s(i+1)) <= pair_tol;

        inOscBand = ((osc_low <= domFreq(i)   && domFreq(i)   <= osc_high) || ...
                     (osc_low <= domFreq(i+1) && domFreq(i+1) <= osc_high));

        if isPair && inOscBand
            grp_osc(i)   = true;
            grp_osc(i+1) = true;
            i = i + 2;
        else
            i = i + 1;
        end
    end

    % Remove any overlap, trend has priority
    grp_osc(grp_trend) = false;

    % Noise: all remaining components
    grp_noise = ~(grp_trend | grp_osc);

    % ---- Reconstruct grouped time series ----
    trend = sum(components(:, grp_trend), 2);
    osc   = sum(components(:, grp_osc),   2);
    noise = sum(components(:, grp_noise), 2);

    % ---- Energy percentages ----
    totalEnergy = sum(x.^2);

    trendEnergyPct = 100 * sum(trend.^2) / totalEnergy;
    oscEnergyPct   = 100 * sum(osc.^2)   / totalEnergy;
    noiseEnergyPct = 100 * sum(noise.^2) / totalEnergy;

    % ---- Reconstruction check ----
    recon = trend + osc + noise;
    reconstructionError = norm(x - recon) / norm(x);

    % ---- Save decomposed components ----
    writematrix(trend, sprintf('trend_%s.csv', outputPrefix));
    writematrix(osc,   sprintf('osc_%s.csv',   outputPrefix));
    writematrix(noise, sprintf('noise_%s.csv', outputPrefix));

    % ---- Save diagnostics table ----
    componentIndex = (1:r).';
    singularValue = s(:);
    dominantFreq = domFreq(:);
    group = strings(r,1);

    group(grp_trend) = "trend";
    group(grp_osc)   = "oscillation";
    group(grp_noise) = "noise";

    diagnosticsTable = table(componentIndex, singularValue, dominantFreq, group);
    writetable(diagnosticsTable, sprintf('diagnostics_%s.csv', outputPrefix));

    % ---- Save summary table ----
    summaryTable = table( ...
        L, ...
        sum(grp_trend), ...
        sum(grp_osc), ...
        sum(grp_noise), ...
        trendEnergyPct, ...
        oscEnergyPct, ...
        noiseEnergyPct, ...
        reconstructionError, ...
        'VariableNames', { ...
        'L', ...
        'NumTrendComponents', ...
        'NumOscillationComponents', ...
        'NumNoiseComponents', ...
        'TrendEnergyPct', ...
        'OscillationEnergyPct', ...
        'NoiseEnergyPct', ...
        'RelativeReconstructionError'} ...
    );

    writetable(summaryTable, sprintf('summary_%s.csv', outputPrefix));

    % ---- Store diagnostics in info struct ----
    info = struct();
    info.L = L;
    info.N = N;
    info.K = K;
    info.grouping_parameters = struct( ...
        'low_freq_thresh', low_freq_thresh, ...
        'pair_tol', pair_tol, ...
        'osc_low', osc_low, ...
        'osc_high', osc_high ...
    );
    info.singular_values = s;
    info.dominant_freqs = domFreq;
    info.groups = struct( ...
        'trend', find(grp_trend), ...
        'oscillation', find(grp_osc), ...
        'noise', find(grp_noise) ...
    );
    info.energy_percent = struct( ...
        'trend', trendEnergyPct, ...
        'oscillation', oscEnergyPct, ...
        'noise', noiseEnergyPct ...
    );
    info.reconstruction_error = reconstructionError;

    % ---- Print summary ----
    fprintf('\nSSD decomposition completed for L = %d\n', L);
    fprintf('Trend components      : %d\n', sum(grp_trend));
    fprintf('Oscillation components: %d\n', sum(grp_osc));
    fprintf('Noise components      : %d\n', sum(grp_noise));
    fprintf('Trend energy          : %.2f %%\n', trendEnergyPct);
    fprintf('Oscillation energy    : %.2f %%\n', oscEnergyPct);
    fprintf('Noise energy          : %.2f %%\n', noiseEnergyPct);
    fprintf('Relative recon. error : %.3e\n\n', reconstructionError);
end

% -------------------------------------------------------------------------
function y = hankel_averaging(Xi)
% Convert LxK elementary matrix back to length-N series by anti-diagonal averaging.

    [L,K] = size(Xi);
    N = L + K - 1;

    y = zeros(N,1);
    cnt = zeros(N,1);

    for i = 1:L
        for j = 1:K
            idx = i + j - 1;
            y(idx) = y(idx) + Xi(i,j);
            cnt(idx) = cnt(idx) + 1;
        end
    end

    y = y ./ max(cnt, 1e-12);
end

% -------------------------------------------------------------------------
function f = dom_freq(x)
% Dominant frequency in cycles/sample using FFT peak, ignoring DC.

    x = x(:) - mean(x);
    N = numel(x);

    if N < 8
        f = 0;
        return;
    end

    X = fft(x);
    M = floor(N/2);

    mag = abs(X(1:M+1));
    mag(1) = 0;   % ignore DC component

    [~, idx] = max(mag);
    f = (idx - 1) / N;
end