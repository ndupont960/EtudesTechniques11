%Projet UE Vision Amaury FUMARD et Paul BLANCHET

%Toutes les fonctions sont à la fin

% Charger l'image
image = imread('A.jpg');

% Convertir en niveaux de gris si nécessaire
if size(image, 3) == 3
    image = rgb2gray(image);
end

% Convertir l'image en double
image = double(image);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%  Méthodes %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Partie 1 : Echelle unique et inhibition %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Calcul du Gradient avec un Noyau Gaussien %%%

%paramètre
sigma = 1;

% Calculer la magnitude du gradient
M_sigma  =  scale_dependent_gradient(image, sigma);

% Afficher le gradient
figure; imshow(M_sigma , []); title('Magnitude du Gradient M_sigma');


%%% Inhibition des Contours Environnants %%%

%paramètre
alpha = 0.005;

% Calcul de l'inhibition
c_sigma = contour_inhibition(M_sigma, sigma, alpha);

% Afficher l'inhibition
figure; imshow(c_sigma , []); title('Inhibition de Contour c_sigma');


%%% Seuillage par hystérésie et suppression des non-maxima %%%

% Seuils à ajuster en fonction de l'image qu'on met en entrée
seuil_bas = 0.0000000000001; 
seuil_haut = 0.1;

b_sigma = seuil_hyst(c_sigma,seuil_bas,seuil_haut);

figure; imshow(b_sigma , []); title('SSCD_sigma');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Partie 2 : Echelle multiple et Canny %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Détection des Contours à l'Échelle Originale avec Canny simga = 1 et 2
% Seuils à ajuster en fonction de l'image qu'on met en entrée
seuil_bas = 0.05; 
seuil_haut = 0.17; 
b1 = edge(image, 'canny', [seuil_bas, seuil_haut],1);
b2 = edge(image, 'canny', [seuil_bas, seuil_haut],2);

figure; imshow(b1); title('b1');
figure; imshow(b2); title('b2');

% Décimer b2
K = 2; % Facteur de décimation
b2 = b2(1:K:end, 1:K:end);

% Dilatation Morphologique
dto = dilat_morph(b2);

% affichache du DTO
figure; imshow(dto); title('DTO');

% Superposition visuelle de b1 et DTO
superposition_visuelle = imfuse(b1, dto);
figure; imshow(superposition_visuelle); title('Superposition Visuelle de b1 et DTO');

% Opération logique AND entre b1 et DTO
b1_and_dto = b1 & dto;
figure; imshow(b1_and_dto); title('b1 AND DTO');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Partie 3 : Echelle multiple et inhibition %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Paramètres
N = 7; % Nombre d'échelles

% Tableau de matrices et calcul des images à plusieurs échelles
images_decimees = cell(N, 1);
images_decimees{1} = image;
for n = 2:N
    images_decimees{n} = decim_im(image,n);
end

% Détection de contours à chaque échelle
contours_binaires = cell(N, 1);
for n = 1:N
    contours_binaires{n} = SingleScaleContourDetector(images_decimees{n});
end

% Combine les contours à différentes échelles : de la plus grossière et en descendant progressivement
resultat_final = contours_binaires{N};
for n = N-1:-1:1
    % Dilatation du résultat actuel
    dto = dilat_morph(contours_binaires{n+1});
    
    % Combinaison avec la carte de contours à l'échelle suivante
    resultat_final = contours_binaires{n} & dto;
    figure;imshow(resultat_final);title('Résultat de la détection de contours multiscale avec inhibition du contour, étape : '+n);
end

imshow(resultat_final);
title('Résultat de la détection de contours multiscale avec inhibition du contour');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%  Fonctions %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Partie 1 : Echelle unique et inhibition %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Calcul du Gradient avec un Noyau Gaussien %%%

function M_sigma = scale_dependent_gradient(image, sigma)
    % Fonction pour calculer le gradient à dépendance d'échelle

    % Création des noyaux gaussiens dérivés
    [Gx, Gy] = gaussian_derivative_kernels(sigma);

    % Convolution de l'image avec les noyaux dérivés
    F_sigma_x = imfilter(image, Gx, 'same', 'replicate');
    F_sigma_y = imfilter(image, Gy, 'same', 'replicate');

    % Calcul de la magnitude du gradient
    M_sigma = sqrt(F_sigma_x.^2 + F_sigma_y.^2);
    
    % Normalisation
    M_sigma = M_sigma / max(M_sigma(:));
end

function [Gx, Gy] = gaussian_derivative_kernels(sigma)
    % Fonction pour créer les noyaux gaussiens dérivés

    % Taille du noyau (un multiple de sigma)
    sz = ceil(sigma * 3) * 2 + 1;

    % Création du noyau gaussien
    [X, Y] = meshgrid(-sz/2 : sz/2, -sz/2 : sz/2);
    G = exp(-(X.^2 + Y.^2) / (2 * sigma^2))/(2 * pi * sigma^2);

    % Dérivées du noyau gaussien
    Gx = -X .* G / (sigma^2);
    Gy = -Y .* G / (sigma^2);
end

%%% Inhibition des Contours Environnants %%%

function dog = difference_of_gaussians(sigma)
    % Taille du noyau (un multiple de sigma)
    sz = ceil(sigma * 3) * 2 + 1;
    
    % Création de la grille de coordonnées
    [X, Y] = meshgrid(-sz/2 : sz/2, -sz/2 : sz/2);

    % Calcul des deux gaussiennes
    G1 = exp(-(X.^2 + Y.^2) / (2 * (sigma * 4)^2));
    G2 = exp(-(X.^2 + Y.^2) / (2 * sigma^2));

    % Difference of Gaussians
    dog = max(G1/(2 * pi * (sigma * 4)^2) - G2/(2 * pi * sigma^2),0);
end

function w_sigma = normalization_term(sigma)
    % Calcul du DoG
    dog = difference_of_gaussians(sigma);

    % Calcul du terme de normalisation
    w_sigma = dog / sum(dog(:).^2);
end

function t_sigma = suppression_term(M_sigma, sigma)
    % Calcul du DoG
    w_sigma = normalization_term(sigma);

    % Convolution du DoG avec la magnitude du gradient
    t_sigma = imfilter(M_sigma, w_sigma, 'same', 'replicate');
end

function c_sigma = contour_inhibition(M_sigma, sigma, alpha)
    % Calcul du terme de suppression
    t_sigma = suppression_term(M_sigma, sigma);

    %supprimer les valeurs négatives
    t_sigma = max(t_sigma,0);

    % Inhibition de contour
    c_sigma =  M_sigma - alpha * t_sigma;
    
    %supprimer les valeurs négatives
    c_sigma = max(c_sigma,0);
end

%%% Seuillage hystérésique et suppression des non-maxima %%%

% suppression des non-maximas

function non_max_suppressed = supp_non_max(c_sigma)

    % Calcul du gradient
    [Fx, Fy] = gradient(double(c_sigma));
    angle = atan2(Fy, Fx);

    % suppression des non-maximas
    non_max_suppressed = zeros(size(c_sigma));
    for i = 2:size(c_sigma, 1)-1
        for j = 2:size(c_sigma, 2)-1
            % Trouver les voisins dans la direction du gradient
          [neighb1, neighb2] = find_neighbors(angle(i, j), i, j);

        % Suppression des non-maxima
            if c_sigma(i, j) >= c_sigma(neighb1(1), neighb1(2)) && c_sigma(i, j) >= c_sigma(neighb2(1), neighb2(2))
                non_max_suppressed(i, j) = c_sigma(i, j);
            end
        end
    end
end 

function b_sigma = seuil_hyst(c_sigma,seuil_bas,seuil_haut)

    % suppression des non-maximas
    non_max_suppressed = supp_non_max(c_sigma);

    b_sigma = zeros(size(c_sigma));
    b_sigma(non_max_suppressed >= seuil_haut) = 1;

    % Propagation des pixels forts aux faibles
    for i = 2:size(c_sigma, 1)-1
        for j = 2:size(c_sigma, 2)-1
            if non_max_suppressed(i, j) >= seuil_bas
                if any(any(b_sigma(i-1:i+1, j-1:j+1) == 1))
                    b_sigma(i, j) = 1;
                end
            end
        end
    end
end

function [neighb1, neighb2] = find_neighbors(angle, i, j)
    % Détermination de la direction du gradient
    if (angle > -pi/8 && angle <= pi/8) || (angle > 7*pi/8 || angle <= -7*pi/8)
        neighb1 = [i, j-1]; neighb2 = [i, j+1];
    elseif (angle > pi/8 && angle <= 3*pi/8) || (angle > -7*pi/8 && angle <= -5*pi/8)
        neighb1 = [i-1, j-1]; neighb2 = [i+1, j+1];
    elseif (angle > 3*pi/8 && angle <= 5*pi/8) || (angle > -5*pi/8 && angle <= -3*pi/8)
        neighb1 = [i-1, j]; neighb2 = [i+1, j];
    elseif (angle > 5*pi/8 && angle <= 7*pi/8) || (angle > -3*pi/8 && angle <= -pi/8)
        neighb1 = [i-1, j+1]; neighb2 = [i+1, j-1];
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Partie 2 : Echelle multiple et Canny %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dilatation morphologique
function dto = dilat_morph(image)

    % Détection des Contours à l'Échelle Originale avec Canny
    b = edge(image, 'canny');

    % Opérateur DTO
    [rows, cols] = find(b);
    beta = zeros(size(b) * 2);
    beta(sub2ind(size(beta), rows * 2, cols * 2)) = 1;

    % Dilatation morphologique
    se = strel('disk', 3);
    dto = imdilate(beta, se);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Partie 3 : Echelle multiple et inhibition %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calcul des images à plusieurs échelles
function im_decimee = decim_im(image,N)
    sigma = 2^N;

    % Taille du noyau (un multiple de sigma)
    sz = ceil(sigma * 3) * 2 + 1;
    % Création du noyau gaussien
    [X, Y] = meshgrid(-sz/2 : sz/2, -sz/2 : sz/2);
    G = exp(-(X.^2 + Y.^2) / (2 * sigma^2))/(2 * pi * sigma^2);

    im_convolu = imfilter(image, G, 'same', 'replicate');

    im_decimee = imresize(im_convolu, 1/2^(N-1), 'bicubic');
end

% SSCD
function SSCD = SingleScaleContourDetector(image)
    
    %%% Calcul du Gradient avec un Noyau Gaussien %%%

    %paramètre
    sigma = 1;

    % Calculer la magnitude du gradient
    M_sigma  =  scale_dependent_gradient(image, sigma);

    %%% Inhibition des Contours Environnants %%%

    %paramètre
    alpha = 0.005;

    % Calcul de l'inhibition
    c_sigma = contour_inhibition(M_sigma, sigma, alpha);

    %%% Seuillage hystérésique et suppression des non-maxima %%%
    seuil_bas = 0.001;  
    seuil_haut = 0.1; 

    SSCD = seuil_hyst(c_sigma,seuil_bas,seuil_haut);

end
