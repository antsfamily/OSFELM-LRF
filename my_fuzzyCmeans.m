function [ U, centers ] = my_fuzzyCmeans( data, n_clusters )
    nRows = size(data, 1);
    nDim = size(data, 2);
    nIteration = 10;
    m = 2;
    
    U = zeros(nRows, n_clusters);
    centers = zeros(n_clusters, nDim);
    
    for i=1:nRows
        U(i,:) = rand(1,n_clusters);
        total = sum(U(i,:));
        U(i,:) = U(i,:)/total;
    end
        
    for itr=1:nIteration
        
        dist = zeros(nRows, n_clusters);
        
        for c=1:n_clusters
            temp1 = U(:,c).^m;
            temp2 = temp1*ones(1,nDim);
            temp2 = data(:,:) .* temp2;
            temp2 = sum( temp2 );
            temp1 = sum( temp1 );
            centers(c,:) = temp2 ./ temp1;
            
            temp1 = data(:,:)-(ones(nRows,1)*centers(c,:));
            temp1 = temp1 .* temp1;
            temp1 = temp1*ones(nDim,1);
            dist(:,c) = sqrt(temp1);
        end
        
        for i=1:nRows
            temp=0;
            for k=1:n_clusters
                temp = temp + 1/( dist(i,k) ^ (2/(m-1)) );
            end
            
            for j=1:n_clusters
                U(i,j) = 1 / ( ( dist(i,j)^(2/(m-1)) )*temp );
            end
        end
        
    end
end

