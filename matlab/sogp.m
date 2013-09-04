function sogp(X, y, X_star)

    global current_size total_count eps_tol capacity s20 l_sq sigmaf_sq
    
    global Q_plot C_plot alpha_plot kstar_plot gamma_plot

    current_size = 0;
    total_count = 0;
    eps_tol = 1e-6;
    capacity = 20;
    s20 = 1e-5;
    l_sq = 10*10;
    sigmaf_sq = 1e-4;
    
    Q_plot = [];
    C_plot = [];
    alpha_plot = [];
    kstar_plot = [];
    gamma_plot = [];
    
    add_measurements(X, y);
    
    xx = linspace(0, 20, 20);
    yy = linspace(0, 20, 20);
    [xx, yy] = meshgrid(xx, yy);
    X_star = [xx(:), yy(:)];
    size(X_star)
    
    plot_variables();
    [f_star, sigconf] = predict_measurements(X_star);
    
    figure
    plot3(X_star(:, 1), X_star(:, 2), f_star, '*')
    hold on
    plot3(X(:, 1), X(:, 2), y, '*r')
end

function plot_variables()

    global Q_plot C_plot alpha_plot kstar_plot gamma_plot
    
    plot(Q_plot); title('Q'); figure
    plot(C_plot); title('C'); figure
    plot(alpha_plot); title('alpha'); figure
    plot(kstar_plot); title('kstar'); figure
    plot(gamma_plot); title('gamma')

end

function add_measurements(X, y)

    for i = 1:size(X, 1)
       add(X(i, :)', y(i)); 
    end

end

function add(X, y)

    global current_size total_count eps_tol capacity s20 alpha C Q BV
    
    global Q_plot C_plot alpha_plot kstar_plot gamma_plot

    total_count = total_count + 1;
    disp(['Adding measurement ' num2str(total_count)])
    kstar = kernel_function(X, X);
    if current_size == 0
        alpha = y / (kstar + s20);
        C = -1 / (kstar + s20);
        Q = 1 / kstar;
        
        current_size = 1;
        BV = X;
    else
        k = construct_covariance(X, BV);
        m = alpha'*k;
        s2 = kstar + k'*C*k;
        
        if s2 < 1e-12
            disp(['s2: ' num2str(s2)]);
        end
        
        r = -1 / (s20 + s2);
        q = -r*(y - m);
        e_hat = Q*k;
        gamma = kstar - k'*e_hat;
        gamma_plot = [gamma_plot gamma];
        if gamma < 1e-12
            disp(['gamma: ' num2str(gamma)]);
            gamma = 0;
        end
        
        if gamma < eps_tol && capacity ~= -1
            disp(['sparse, gamma: ' num2str(gamma)]);
            eta = 1 / (1 + gamma*r);
            s_hat = C*k + e_hat;
            alpha = alpha + s_hat*(q * eta);
            C = C + r*eta*(s_hat*s_hat');
        else
            disp('full');
            s = [C*k; 1];
            alpha = [alpha; 0] + q*s;
            C = [[C zeros(size(C, 1), 1)]; zeros(1, size(C, 2)+1)] + r*(s*s');
            BV = [BV X];
            current_size = current_size + 1;
            Q = [[Q zeros(size(Q, 1), 1)]; zeros(1, size(Q, 2)+1)];
            e_hat = [e_hat; -1];
            Q = Q + 1/gamma*(e_hat*e_hat');
        end
        
        while current_size > capacity && capacity > 0
           minscore = 0;
           minloc = -1;
           for i = 1:current_size
              score = alpha(i)^2/(Q(i, i) + C(i, i));
              if i == 1 || score < minscore
                  minscore = score;
                  minloc = i;
              end
           end
           delete_bv(minloc);
           disp(['deleting for size: ' num2str(current_size)]);
        end
        
        minscore = 0;
        minloc = -1;
        while minscore < 1e-9 && current_size > 1
           for i = 1:current_size
               score = 1/Q(i, i);
               if i == 1 || score < minscore
                   minscore = score;
                   minloc = i;
               end
           end
           if minscore < 1e-9
               disp('deleting for geometry');
               delete_bv(minloc);
           end
        end
        
        if isnan(C(1, 1))
            disp('C is nan');
        end
    end
    
    Q_plot = [Q_plot Q(1, 1)];
    C_plot = [C_plot C(1, 1)];
    alpha_plot = [alpha_plot alpha(1, 1)];
    kstar_plot = [kstar_plot kstar];
    
end

function delete_bv(loc)

    global current_size alpha C Q BV

    alphastar = alpha(loc);
    alpha(loc) = alpha(end);
    alpha = alpha(1:end-1);
    
    cstar = C(loc, loc);
    Cstar = C(:, loc);
    Cstar(loc) = Cstar(end);
    Cstar = Cstar(1:end-1);
    
    Crep = C(:, end);
    Crep(loc) = Crep(end);
    C(loc, :) = Crep';
    C(:, loc) = Crep;
    C = C(1:end-1, 1:end-1);
    
    qstar = Q(loc, loc);
    Qstar = Q(:, loc);
    Qstar(loc) = Qstar(end);
    Qstar = Qstar(1:end-1);
    Qrep = Q(:, end);
    Qrep(loc) = Qrep(end);
    Q(loc, :) = Qrep';
    Q(:, loc) = Qrep;
    Q = Q(1:end-1, 1:end-1);
    
    alpha = alpha - alphastar / (qstar + cstar) * (Qstar + Cstar);
    C = C + (Qstar * Qstar')/qstar - ((Qstar+Cstar)*(Qstar+Cstar)')/(qstar+cstar);
    Q = Q - (Qstar * Qstar')/qstar;
    
    BV(:, loc) = BV(:, end);
    BV = BV(:, 1:end-1);
    
    current_size = current_size - 1;

end

function [f_star, sigconf] = predict_measurements(X_star)

    f_star = zeros(size(X_star, 1), 1);
    sigconf = zeros(size(X_star, 1), 1);
    for c = 1:size(X_star, 1)
        [f, s] = predict(X_star(c, :)');
        f_star(c) = f;
        sigconf(c) = s;
    end

end

function [f_star, sigma] = predict(X_star)

    global current_size alpha C BV s20

    kstar = kernel_function(X_star, X_star);
    k = construct_covariance(X_star, BV);
    f_star = 0;
    
    if current_size == 0
        sigma = kstar + s20;
    else
        f_star = alpha'*k;
        sigma = s20 + kstar + k'*C*k;
    end
    
    if sigma < 0
        disp('sigma < 0!');
        sigma = 0;
    end

end

function K = construct_covariance(X, Xv)

    K = zeros(size(Xv, 2), 1);
    for i = 1:size(Xv, 2)
        K(i) = kernel_function(X, Xv(:, i));
    end

end

function k = kernel_function(xi, xj)

    global l_sq sigmaf_sq

    k = sigmaf_sq*exp(-0.5 / l_sq * norm(xi - xj).^2);

end