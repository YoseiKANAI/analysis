DFA_Raw_103 = cumsum(database.CoP_y(2:end,1)); % 累積和を出してる？
NN = length(database.CoP_y(2:end,1)); % データの長さを取得
n_min = 10;
n_max = NN/4;
scales = n_min:10:n_max;
n_vals = length(scales);
for j = 1:n_vals % スケール数分
        n = scales(j);
    y = removetrend(DFA_Raw_103, n); % Detrend the time series for the given scale
    F(j) = sqrt(mean(y.^2)); % Calculate fluctuation function
end
logscales = log10(scales);
logF = log10(F);
P_All_Raw_100 = polyfit(logscales, logF,1);
database.Alph_All_Raw_100 = P_All_Raw_100(1,1);


function y = removetrend(x, n)
    N = length(x);
    n_segments = floor(N/n); % Number of segments
    y = zeros(N, 1);
    for i = 1:n_segments
        start_idx = (i-1)*n + 1;
        end_idx = i*n;
        segment = x(start_idx:end_idx);
        t = (1:n)';
        p = polyfit(t, segment, 1); % Least-squares linear regression
        trend = polyval(p, t); % Trend for the segment
        y(start_idx:end_idx) = segment - trend; % Remove trend
    end
end