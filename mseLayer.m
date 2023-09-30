
classdef mseLayer < nnet.layer.RegressionLayer
    
    % This version has output weighting.
        
    properties
        % (Optional) Layer properties.
        
        output_weight

        % Layer properties go here.
    end
 
    methods
        
        function layer = mseLayer(varargin)
            
            p = inputParser;
            addParameter(p, 'output_weight', 1, @isrow)
            parse(p, varargin{:})
            
            layer.output_weight = p.Results.output_weight;
            
        end
        
        function loss = forwardLoss(layer, Y, T)
            
            w = layer.output_weight;
            n = size(Y,2);
            
            U = Y - T;
            
            i = isinf(T);
            U(i) = 0;
            
            Q = 0.5*U.^2.*w';
            
            loss = sum(Q, 'all')/n;
            
        end
        
        function dLdY = backwardLoss(layer, Y, T) 
            
            w = layer.output_weight;
            n = size(Y,2);
            
            U = Y - T;
            
            i = isinf(T);
            U(i) = 0;
            
            dLdY = U.*w'/n;
            
        end
    end
end

% function loss = forwardLoss(layer, Y, T)
% 
% o = layer.output_weight;
% n = size(Y,2);
% k = size(Y,1)/2;
% 
% W = T(k+1:2*k, :).*o';
% U = Y(1:k, :) - T(1:k, :);
% 
% Q = 0.5*U.^2.*W;
% 
% loss = sum(Q, 'all')/n;
% 
% end
% 
% function dLdY = backwardLoss(layer, Y, T)
% 
% o = layer.output_weight;
% n = size(Y,2);
% k = size(Y,1)/2;
% 
% W = T(k+1:2*k, :).*o';
% U = Y(1:k, :) - T(1:k, :);
% 
% dLdU = U.*W/n;
% 
% dLdY = [dLdU; zeros(k,n)];
% 
% end
