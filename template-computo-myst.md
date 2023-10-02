---
title: "Point process discrimination according to repulsion"
# subtitle: "Example based on the myst system"
author: "Hamza ADRAT and Laurent DECREUSEFOND"
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Abstract
In numerous applications, cloud of points do seem to exhibit *repulsion* in the intuitive sense that there is no local cluster as in a Poisson process. Motivated by data coming from cellular networks, we devise a classification algorithm based on the form of the Voronoi cells. We show that, in the particular set of data we are given, we can retrieve some repulsiveness between antennas, which was expected for engineering reasons.

# Introduction
In the performance analysis of cellular systems, the locations of antennas (or base stations) play a major role (see {cite}`BaccelliStochasticGeometryWireless2008`). It is usually admitted that they can be modeled by a Poisson process. But the data which can be gathered from the Web site of the French National Agency of Radio Frequencies, Cartoradio, see {cite}`ANFR`, tend to prove that this may not be the case. More precisely, if we look at the global picture of all antennas in Paris, we see features reminiscent of a Poisson process (local clusters for instance), see {numref}`paris-orange-fig` (left). However, if we look closer and finer, by specifying a region and a frequency band, we see that the antennas locations do seem to exhibit some repulsion (see {numref}`paris-orange-fig`, right picture).

```{figure} /paris-orange.png
---
name: paris-orange-fig
---
Left: Antennas in Paris. Right: Antennas in one frequency  band only.

```

In previous papers, point processes with repulsion have been used to model such systems {cite}`Deng2014`, {cite}`Miyoshi2016`, {cite}`Gomez2015` for no reason but a mere resemblance between the pictures like the right picture in {numref}`paris-orange-fig` and those obtained by simulating a point process with repulsion. The question is then to decide, given one sample of positions of base stations in a bounded domain, whether it is more likely to be modeled by a point process with repulsion or by a *neutral* point process, i.e. where the locations could be considered as coming from independent drawings of some identically distributed random variables. As we only have a single realization,  we cannot use frequency methods. Since the observation window is finite, we cannot either resort to estimates based on stationarity or ergodicity and  we must take care from the side effects.

The rationale behind our work comes from {cite}`goldman_palm_2010`. It is shown there that the Voronoi cells of the Ginibre point process (a particular point
process with repulsion, see below for the exact definition) are in some sense more regular (closer to a circle) than those of a Poisson process (see {eq}`theorem_goldman` in Theorem 1.). By simulation, this feature seems to persist for other point processes with repulsion, like Gibbs processes. In {cite}`Taylor2012`, the surface of Voronoi cells is claimed to be a good discrepancy indicator between Poisson process and several processes with repulsion (Gibbs processes, Strauss processes with repulsion and the Geyer saturation model). For any of these models, we do not have any closed formula on the surface of the Voronoi cells so the procedure proposed in this paper is to simulate a large number of realizations of each of these processes and compute the empirical mean and variance of the Voronoi cells area. They obtain mixed conclusions as this sole indicator does not enable to rule out the Poisson hypothesis for many situations.

Our contribution is to consider the ratio of the surface by the squared perimeter instead of the surface of the Voronoi cells alone. Actually, we can interpret
the result of {cite}`goldman_palm_2010` by saying that the Voronoi cells of a Ginibre point process are more circular than those of a Poisson point process. The isoperimetric inequality stands for any regular enough domain in the plane, $R = \frac{4 \pi S}{P^2}$ is less than $1$ and the equality is obtained for disks. It is thus sensible to think that the ratio $R$ will be closer to $1$ for repulsive processes than for neutral point processes. Following the procedure of {cite}`Taylor2012`, we show that we get a much better indicator by using $R$ instead $S$ alone to discriminate between repulsive and neutral point processes.

However, for the application we have in mind, which is to decide for one single map  which model is the most pertinent, we cannot use this criterion based on probability. That is why we resort to an ML model. After several tries, we concluded that the most efficient algorithm was to use Logistic Regression. In a first step, we trained it on simulations of Ginibre and Poisson point processes. The advantage of the Ginibre process is that we have efficient algorithm to simulate it {cite}`MR4279876` and it does not seem to alter the accuracy of our algorithm to use one single class of repulsive point process. We remarked that we obtain a much better discrimination by considering the mean value of $R$ for the five most central cells instead of just the most central one. We can even improve our discrimination rate by adding to the input vector the value of each of the five ratios.

Furthermore, the repulsion in the Ginibre class of point processes can be also modulated by making a $\beta$-thinning (to weaken the repulsion) and then a $\sqrt{\beta}$-dilation (to keep the same intensity of points per surface unit) to obtain what is called a $\beta$-Ginibre. For $\beta=1$, we have the original Ginibre process and when $\beta$ goes to $0$, it tends in law to a Poisson process (see {cite}`DecreusefondAsymptoticssuperpositionpoint2015`) so that we have a full scale of point processes with intermediate repulsion between $0$ and $1$. We show that our logistic regression algorithm can still accurately discriminate between Poisson and $\beta$-repulsive point processes for $\beta$ up to $0.7$.

The paper is organized as follows. We first remind what is a Ginibre point process and the property of its Voronoi cells which motivates the sequel.

# Preliminaries

We consider finite point processes on a bounded window $E$. The law of a such a point process $N$ can be  characterized by its correlation functions (for
details we refer to {cite}`Daley2003`[Chapter 5]). These are symmetric functions $(\rho_{k},k\ge 1)$ such that for any bounded function $f$, we can write:

$$
 \mathbb{E}\left[ \sum_{\alpha \subset N} f(\alpha) \right] = \sum_{k=1}^{+ \infty} \frac{1}{k!} \int_{E^k} f(\{x_1, \dots, x_k\}) \rho_{k}(x_1, \dots, x_k) \, d x_1 \dots d x_k .
$$

Intuitively speaking, $\rho_{k}(x_{1}, \dots, x_{k}) \, d x_{1} \dots d x_{k}$ represents the probability to observe in $N$, at least $k$ points located around the 
point $x_{j}$. For a Poisson point process of control measure $m(x) \, dx$, we have

$$
\rho_{k}(x_{1}, \dots, x_{k}) = \prod_{j=1}^{k} m(x_{j}).
$$

The **Ginibre point process**, restricted to $E=B(0,r)$, with intensity $\rho = \frac{\lambda}{\pi}$ (with $\lambda > 0$) has correlation functions (see {cite}`Decreusefond_2015`)

```{math}
:label: correlation_functions_determinantal
\rho_{k}(x_1, \dots, x_k) = \det(K(x_i, x_j), \; 1 \le i,j \le k)
```
where $K$ is given by

```{math}
:label: eq_main
K_r(x,y)=\sum_{j=1}^\infty \frac{\gamma(j+1,r^2)}{j!} \phi_j(x)\phi_j(\bar y)
```
with

$$
\phi_j(x) = \sqrt{\frac{\rho}{\gamma(j+1,r^2)}} \left(\sqrt{\lambda} x \right)^j \, e^{-\frac{\lambda}{2} |x|^2}
$$

and $\gamma(n,x)$ is the lower incomplete Gamma function. The simulation of such a point process is a delicate matter, first solved in {cite}`HoughDeterminantalprocessesindependence2006`. It remains costly because the algorithm contains complex calculations and some rejections. In order to fasten the procedure, an approximate algorithm, with error estimates, has been given in {cite}`MR4279876` (see the bibliography therein to get the URL of the Python code).

For an at most denumerable set of points $\{x_{n}, \, n \ge 1\}$, the Voronoi cells are defined as the convex sets

$$
\mathcal{C}(x_{i})=\{z \in \mathbb{C},\ |z-x_{i}|\le |z-x_{j}|  \text{ for all }j\neq i\}.
$$

When the points are drawn from a point process, we thus have a collection of random closed sets. When the process under consideration is stationary with respect to translations, it is customary to define the typical law of a Voronoi cell as the law of the cell containing the origin of $\mathbb{R}^{2}$ when the point process is taken under its Palm distribution {cite}`goldman_palm_2010`, {cite}`BaccelliStochasticGeometryWireless2009`. It turns out that we know the Palm distribution of the Poisson process (which is itself) and of the Ginibre point process (the correlation functions are of the form {eq}`correlation_functions_determinantal` with $K$ being $K_{R}$ with the first term removed). 
We denote by $\mathcal{C}_p$ (respectively $\mathcal{C}_{G}$) the typical cell of the Voronoi tessellation associated to a stationary Poisson process in $\mathbb{C}$  with
intensity $\lambda$ (respectively to the Ginibre point process of intensity $\rho$). One of the main theorems of {cite}`goldman_palm_2010` is the following.

**Theorem 1.**
When $r \to 0,$
```{math}
:label: theorem_goldman
\mathbb{E} \left[ V(\mathcal{C}_{G} \cap B(0,r)) \right] = \mathbb{E} \left[ V(\mathcal{C}_p \cap B(0,r)) \right] (1 + r^2 W + \circ(r^2))
```
where $W$ is a positive random variable.

This theorem shows that near the germs of the cells a more important part of the area is captured in the Ginibre–Voronoi tessellation than in the Poisson–Voronoi tessellation. This is an indication that the Voronoi cells of the Ginibre point process are more circular than those given by the Poisson process. This can be corroborated by simulation as shows the Figure {numref}`voronoi-fig`

```{figure} /Voronoi.png
---
name: voronoi-fig
---
On the left, Voronoi cells associated to a realization of a Ginibre process. On the right, Voronoi cells associated to a realization of a Poisson process.
```

As we know that circles saturate the isoperimetric inequality, it is sensible to consider classification algorithms based on area and squared perimeter of Voronoi cells. In order to avoid side effects, we concentrate on the innermost cells of the observation window.

# Classification of CARTORADIO data

The Cartoradio web site contains the locations (in GPS coordinates) and other informations about all the antennas (or base stations) in metropolitan France for any operator, any frequency band and all generation of wireless systems (2G to 5G). The capacity of an antenna depends on its power and on the traffic demand it has to serve.  Outside metropolitan areas, the antennas are relatively scarce and located along the main roads to guarantee a large surface coverage (around 30 km$^2$). Hence there is no need to  construct models for these regions.  On the contrary, in big towns, the density of base stations is much higher to handle the traffic demand: An antenna covers around half a squared kilometer. This is  where the dimensioning problem do appear. One should have a sufficient number of antennas per unit of surface to transport all the traffic, on the other hand, base stations operating in a given frequency band cannot be to close to mitigate interference. This explains the right picture of Figure {numref}`paris-orange-fig`.

When it comes to assess the type of point process we should consider in this situation, we cannot consider the city as a whole: the geography (notably the Seine river in Paris, the parks, etc.), the non uniformity of demands (the traffic is heavier aroung railway stations or touristic sites,  for instance) which entails a higher density of antennas,  ruin any kind of invariance a statistician  could hope for. That means, we should restrict our expectations to local models of  the size of a district or a bit more. Since interference, which are the main annoyance to be dealt with, are a local phenomenon, working on a partial part of the whole domain is sufficient to predict the behavior and dimension a wireless network.

In the following sections, we will use Python code that assumes that the following packages have been loaded:

```{code-cell} ipython3
:tags: [hide-output, hide-input]

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import bernoulli
from scipy.spatial import Voronoi, ConvexHull
import matplotlib.pyplot as plt
import seaborn as sns
font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 11,}
```

## Statistical approach
Given a circular domain with $N$ points, we want to decide whether the points exhibit repulsion or not. To do so, we will begin with a statistical approach, where we will first calculate, for Poisson processes as well as for Ginibres and $\beta$-Ginibres processes, the probability that the ratio $R = \frac{4 \pi S}{P^2}$ of the central cell is less than or equal to $r$, for values of $r$ ranging from $0$ to $1$. And then we will apply the same approach using the mean ratio of the five central cells. Finally, we will calculate $95$% confidence intervals for each of these processes.

The following code illustrates the generation of various point samples and the calculation of ratios by defining the number of points $N$ and the parameter $\beta$ for $\beta$-Ginibre processes.

```{code-cell} ipython3
:tags: [show-output, hide-input]

def in_box(towers, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0], towers[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= towers[:, 1], towers[:, 1] <= bounding_box[3]))


def voronoi(towers, bounding_box, N):
    # Select towers inside the bounding box
    i = in_box(towers, bounding_box)
    # Mirror points
    points_center = towers[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left, points_right, axis=0),
                                 np.append(points_down, points_up, axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = Voronoi(points)
    # Filter regions
    # regions = []
    # [vor.point_region[i] for i in range(N)]

    vor.filtered_points = points_center
    vor.filtered_regions = [vor.regions[vor.point_region[i]] for i in range(len(points_center))]
    return vor

def central_area_perim(vor):  
    return ConvexHull(vor.vertices[vor.filtered_regions[0], :]).volume, ConvexHull(vor.vertices[vor.filtered_regions[0], :]).area

def area_perim(vor):
    area, perimeter = [], []
    for i in range(5):
        if len(vor.filtered_regions) >= i:
            area.append(ConvexHull(vor.vertices[vor.filtered_regions[i], :]).volume)
            perimeter.append(ConvexHull(vor.vertices[vor.filtered_regions[i], :]).area)
        else:
            area.append(np.mean(area))
            perimeter.append(np.mean(perimeter))
    return area, perimeter

def ginibre(N, cells):
    radius = (np.sqrt(N)) ; precision = 2**-53 ; error = False ; quiet=True ; output=None 
    args = [radius, N, kernels['ginibre'], precision, error, quiet, output]
    
    sample_ginibre = sample(*args)
    X_ginibre, Y_ginibre = sample_ginibre.real, sample_ginibre.imag
    
    ginibre_points = np.array([X_ginibre, Y_ginibre]).T
    indices = np.argsort((ginibre_points[:,0])**2 + ((ginibre_points[:,1])**2))
    ginibre_points = ginibre_points[indices]
    ginibre_vor = voronoi(ginibre_points, (-np.sqrt(N)-.1, np.sqrt(N)+.1, -np.sqrt(N)-.1, np.sqrt(N)+.1), len(ginibre_points))
    
    if cells == 1:
        vor_area, vor_perim = central_area_perim(ginibre_vor)
    else:
        vor_area, vor_perim = area_perim(ginibre_vor)
    
    return vor_area, vor_perim

def beta_ginibre(N, beta, cells):
    radius = (np.sqrt(N)) ; precision = 2**-53 ; error = False ; quiet=True ; output=None 
    args = [radius, N, kernels['ginibre'], precision, error, quiet, output]
    
    sample_beta_ginibre = (sample(*args))*(bernoulli.rvs(beta, size=N))
    sample_beta_ginibre = np.array([a for a in sample_beta_ginibre if a != 0])*(np.sqrt(beta))
    X_beta_ginibre, Y_beta_ginibre = sample_beta_ginibre.real, sample_beta_ginibre.imag
    
    beta_ginibre_points = np.array([X_beta_ginibre, Y_beta_ginibre]).T
    indices = np.argsort((beta_ginibre_points[:,0])**2 + ((beta_ginibre_points[:,1])**2))
    beta_ginibre_points = beta_ginibre_points[indices]
    
    beta_ginibre_vor = voronoi(beta_ginibre_points, 
                               (-np.sqrt(N*beta)-.1, np.sqrt(N*beta)+.1, -np.sqrt(N*beta)-.1, np.sqrt(N*beta)+.1), 
                               len(beta_ginibre_points))
    
    if cells == 1:
        vor_area, vor_perim = central_area_perim(beta_ginibre_vor)
    else:
        vor_area, vor_perim = area_perim(beta_ginibre_vor)
    
    return vor_area, vor_perim

def poisson(N, cells):
    radius = np.sqrt(N)
    alpha = 2 * np.pi * scipy.stats.uniform.rvs(0,1,N)
    r = radius * np.sqrt(scipy.stats.uniform.rvs(0,1,N))
    
    X_poisson, Y_poisson = r*np.cos(alpha), r*np.sin(alpha)
    poisson_points = np.array([X_poisson, Y_poisson]).T
    
    indices = np.argsort((poisson_points[:,0])**2 + ((poisson_points[:,1])**2))
    poisson_points = poisson_points[indices]
    poisson_vor = voronoi(poisson_points, (-radius -.1, radius +.1, -radius -.1, radius +.1), len(poisson_points))
    
    if cells == 1:
        vor_area, vor_perim = central_area_perim(poisson_vor)
    else:
        vor_area, vor_perim = area_perim(poisson_vor)
        
    return vor_area, vor_perim

def ratio_ginibre(N, cells):
    G = ginibre(N, cells)
    return np.mean(4*np.pi*np.array(G)[0]/(np.array(G)[1])**2)

def ratio_beta_ginibre(N, beta, cells):
    beta_G = beta_ginibre(N, beta, cells)
    return np.mean(4*np.pi*np.array(beta_G)[0]/(np.array(beta_G)[1])**2)

def ratio_poisson(N, cells):
    P = poisson(N, cells)
    return np.mean(4*np.pi*np.array(P)[0]/(np.array(P)[1])**2)

%run -i Moroz_dpp.py
```

The simulation algorithm, as presented in Figure ..., provides a method for computing the quantity $\mathbb{P} \left( \frac{4 \pi A}{P^2} \le r \right)$ as a function of $r$ for the Ginibres processes (the same algorithm is applied to other processes as well). The Algorithm takes as input the number of points $N$, the number of experiences for the simulation $N_{exp}$ and the range of the varibale $r$ as a list of values. Since the simulations require a lot of time to run, we are not going to attach the associated python code, the latter is based on the algorithm described previously.

Figure {numref}`simulation-fig` shows the results of the simulations, where we compare the confidence intervals of the poisson process and the Ginibre process, using first the central cell and then the five central cells. 

```{figure} /simulation.png
---
name: simulation-fig
---
Simulation results using the central cell (up) and the five central cells (down).
```




### Model training

Given a circular domain with $N$ points, we want to decide whether the points exhibit  repulsion or not. Since the repulsion is not sensitive to scaling, we normalize the radius to $R=\sqrt{N}$. This is due to the fact that a cloud drawn from a  Ginibre point process of intensity $1$  with $N$ points occupies roughly a disk with this radius. We train our models on datas issued from drawings of Ginibre configuration and from drawings of $N$ points independently and uniformly scattered in $B(0,\sqrt{N})$.

For each of the previous samples, we compute the Voronoi diagrams and retain only the $10$ cells which are the closest to the barycenter of the configuration in order to avoid side effects. We then  extract the areas and the perimeters  of these cells. In principle, this should be enough to have a good classification algorithm. We noticed that in addition to thes quantities, it dramatically improves the results of the classification of we add  the mean of the first $5$, $10$, $15$ and $20$ cells' areas in order to have a more global information.

The final data will be a set of observations where each one contains $29$ columns described as follow:
- For $1 \le i \le 10, \; \mathrm{V}_i$ is the area of the $i^{\mathrm{th}}$ Voronoi cell.
- For $1 \le j \le 10, \; \mathrm{P}_j$ is the square perimeter of the $j^{\mathrm{th}}$ Voronoi cell.
- For $k \in \{5, 10, 15, 20\}, \; \mathrm{MV}_{k}$ (respectively $\mathrm{MP}_{k}$) is the average area (respectively square perimeter) of the first $k$ Voronoi cells.
- A binary column that represent the type of the initial configuration ($1$ for repulsive/Ginibre - $0$ for non-repulsive/Poisson).

The final column will be the target variable for our classification models.

```{code-cell} ipython3
:tags: [show-output, show-input]

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import Delaunay, Voronoi, ConvexHull

font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 11,}

# Useful functions for creating the data:

def convert_complex_points(l):
    return l.real, l.imag

def convert_lists_to_points(l1, l2):
    return np.array([l1, l2]).T

def extract_Voronoi_areas(vor):  
    areas= []
    perim = []
    for i in range(len(vor.filtered_regions)):
        areas.append(round(ConvexHull(vor.vertices[vor.filtered_regions[i], :]).volume, 2))
        perim.append(round(ConvexHull(vor.vertices[vor.filtered_regions[i], :]).area, 2))
    return areas, list(np.around((np.array(perim))**2, 2))

def in_box(towers, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0], towers[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= towers[:, 1], towers[:, 1] <= bounding_box[3]))


def voronoi(towers, bounding_box, N):
    # Select towers inside the bounding box
    i = in_box(towers, bounding_box)
    # Mirror points
    points_center = towers[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left, points_right, axis=0),
                                 np.append(points_down, points_up, axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = Voronoi(points)
    # Filter regions
    # regions = []
    # [vor.point_region[i] for i in range(N)]

    vor.filtered_points = points_center
    vor.filtered_regions = [vor.regions[vor.point_region[i]] for i in range(len(points_center))]
    return vor

def dpp_Moroz(N):
    radius = int(np.sqrt(N)) ; precision = 2**-53 ; error = False ; quiet=True ; output=None 
    args = [radius, N, kernels['ginibre'], precision, error, quiet, output]
    
    moroz_dpp = sample(*args)
    X_dpp_Mz, Y_dpp_Mz = convert_complex_points(moroz_dpp)
    X_dpp_Mz = X_dpp_Mz*((np.sqrt(N))/radius) ; Y_dpp_Mz = Y_dpp_Mz*((np.sqrt(N))/radius)
    dpp_Mz_points = convert_lists_to_points(X_dpp_Mz, Y_dpp_Mz)
    
    indices = np.argsort((dpp_Mz_points[:,0])**2 + ((dpp_Mz_points[:,1])**2))
    dpp_Mz_points = dpp_Mz_points[indices]
    
    dpp_Mz_vor = voronoi(dpp_Mz_points, (-np.sqrt(N)-.1, np.sqrt(N)+.1, -np.sqrt(N)-.1, np.sqrt(N)+.1), len(dpp_Mz_points))
    Voronoi_areas, Voronoi_perim = extract_Voronoi_areas(dpp_Mz_vor)
    
    return [Voronoi_areas, Voronoi_perim, 1]

def random_process(N):
    radius = np.sqrt(N)
    alpha = 2*np.pi*scipy.stats.uniform.rvs(0,1,N)
    r = radius*np.sqrt(scipy.stats.uniform.rvs(0,1,N))
    
    X_rand, Y_rand = r*np.cos(alpha), r*np.sin(alpha)
    rand_points = convert_lists_to_points(X_rand, Y_rand)
    
    indices = np.argsort((rand_points[:,0])**2 + ((rand_points[:,1])**2))
    rand_points = rand_points[indices]
    
    rand_vor = voronoi(rand_points, (-radius -.1, radius +.1, -radius -.1, radius +.1), len(rand_points))
    Voronoi_areas, Voronoi_perim = extract_Voronoi_areas(rand_vor)
    
    return [Voronoi_areas, Voronoi_perim, 0]

def create_dataframe(N, observations):
    list_df = []
    for i in range(observations):
        list_df.append(dpp_Moroz(N))
        list_df.append(random_process(N))
    df = pd.DataFrame(list_df, columns =['Voronoi_areas', 'Voronoi_perim', 'Type'])
    return df

def normalize(vec):
    vec = np.array(vec)
    m, e = np.mean(vec), np.sqrt(np.var(vec))
    return (vec - m)/e

def compute_mean(l, n):
    if n <= len(l):
        new_l = np.array(list(list(zip(*l))[:n])).T
    else:
        new_l = np.array(list(list(zip(*l))[:len(l)])).T

    return np.mean(new_l, axis=1)

def single_area(l, k):
    l = np.array(l)
    return l[:,k]

def transform_df(odf):
    
    list_V = odf['Voronoi_areas'].tolist()
    list_P = odf['Voronoi_perim'].tolist()
    
    [MV5, MV10, MV15, MV20] = [compute_mean(list_V, n) for n in [5, 10, 15, 20]]
    [MP5, MP10, MP15, MP20] = [compute_mean(list_P, n) for n in [5, 10, 15, 20]]
    
    normalized_V10 = [normalize(list_V[i][:10]) for i in range(odf.shape[0])]
    normalized_P10 = [normalize(list_P[i][:10]) for i in range(odf.shape[0])]
    
    [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10] = [single_area(normalized_V10, k) for k in range(10)]
    [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10] = [single_area(normalized_P10, k) for k in range(10)]
    
    dict_df = {'V1':V1, 'V2':V2, 'V3':V3, 'V4':V4, 'V5':V5, 'V6':V6, 'V7':V7, 'V8':V8, 'V9':V9, 'V10':V10,
               'MV5':MV5, 'MV10':MV10, 'MV15':MV15, 'MV20':MV20,
               'P1':P1, 'P2':P2, 'P3':P3, 'P4':P4, 'P5':P5, 'P6':P6, 'P7':P7, 'P8':P8, 'P9':P9, 'P10':P10,
               'MP5':MP5, 'MP10':MP10, 'MP15':MP15, 'MP20':MP20,
               'type': odf['Type']}
    
    return pd.DataFrame(dict_df)

%run -i Moroz_dpp.py
```

Here is an example of the data created. We generate a data of $2,000$ observations of configurations ($1,000$ repulsive and $1,000$ non repulsive) of $N = 24$ points.

```{code-cell} ipython3
:tags: [show-output, show-input]

ddf = create_dataframe(24, 1000)
ddf_transformed = transform_df(ddf)
ddf_transformed.head()
```

### Classification models

In this section, we  train and test some Machine Learning models using the data we've created in the previous section. For a start we  select all the columns as inputs to our models (this can lead to false predictions, especially if some columns share the same information). Note that  we  only use baseline models, i.e. all the hyperparameters' values are taken as defaults, (a grid search can be used later in order to select the optimal hyperparameters for each model).

The models we use in this paper are:
- Logistic regression, which is a classification algorithm, used when the value of the target variable is categorical in nature. Logistic regression is most commonly used when the data in question has binary output, so when it belongs to one class or another, or is either a $0$ or $1$.
- Random forest, a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the accuracy and control over-fitting.

Two other classification models (Support Vector Machine and XGBoost) have been tested but did not yield more significant results than those two models.

```{code-cell} ipython3
:tags: [show-output, show-input]

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# Useful function for evaluating our models:

def model_Evaluate(model, x_tt, y_tt):
    y_pred = model.predict(x_tt)
    print(classification_report(y_tt, y_pred))
    
    cf_matrix = confusion_matrix(y_tt, y_pred)
    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    logit_roc_auc = roc_auc_score(y_tt, model.predict(x_tt))
    fpr, tpr, thresholds = roc_curve(y_tt, model.predict_proba(x_tt)[:,1])
    
    fig = plt.figure(figsize=(12, 5))
    # Adds subplot on position 1
    ax = fig.add_subplot(121)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '', xticklabels = categories, yticklabels = categories)
    ax.set_title("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    ax.set(xlabel='Predicted values', ylabel='Actual values')

    # Adds subplot on position 2
    ax = fig.add_subplot(122)
    ax.plot(fpr, tpr, label='area = %0.2f' % logit_roc_auc)
    ax.plot([0, 1], [0, 1],'r--', label='Standard')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.0, 1.05])
    
    thresholds_rounded = [round(num, 1) for num in thresholds]
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if threshold in thresholds_rounded:
            index = thresholds_rounded.index(threshold)
            ax.annotate(threshold, (fpr[index], tpr[index]))

    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.set(xlabel='False Positive Rate (1-specificity)', ylabel='True Positive Rate (sensitivity)')
    ax.legend(loc="lower right")
    ax.grid()
    plt.show()


model_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'MV5', 'MV10', 'MV15', 'MV20',
             'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'MP5', 'MP10', 'MP15', 'MP20']
X = ddf_transformed[model_cols].values
y = ddf_transformed['type'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle= True, random_state=7)
```

Here are the results of the logistic regression

```{code-cell} ipython3
:tags: [show-output, show-input]

Baseline_LR = make_pipeline(StandardScaler(), LogisticRegression())
Baseline_LR.fit(X_train, y_train)
model_Evaluate(Baseline_LR, X_test, y_test)
```

Here are the results of the random forest

```{code-cell} ipython3
:tags: [show-output, show-input]

baseline_RF = RandomForestClassifier()
baseline_RF.fit(X_train,y_train)
model_Evaluate(baseline_RF, X_test, y_test)
```

### Testing on CARTORADIO data

CARTORADIO data is a set of configurations of some mobile phone base stations in Paris. The goal is to decide from the classification models already used, whether the configuration do present some repulsion.

The initial data (positions of the  antennas) cover a large area of the city of Paris (see {numref}`cartoradio-fig`(right))
With a real dataset,  we often encounter the problem of heterogeneity between the different parts of the configurations since they depend on the topology  of the space in which the antennas are placed.

To cope with this problem, we  extract from each configuration a representative sample similar to the type of training data so that the tests make sense. {numref}`cartoradio-fig` (left) shows a sample extracted from a given configuration.

```{figure} /cartoradio.png
---
name: cartoradio-fig
---
On the left, Initial cartoradio configuration. On the right, Sample extracted from it and scaled.
```

In the following, we  read the CARTORADIA directly from the "data_cartoradio.csv" file.

```{code-cell} ipython3
# Useful functions to transform the CARTORADIO data

def update_df(odf): 
    converted_V = odf['Voronoi_areas'].str[1:-1].str.split(',').tolist()
    converted_P = odf['Voronoi_perim'].str[1:-1].str.split(',').tolist()
    
    list_V = [(np.float_(converted_V[i])).tolist() for i in range(odf.shape[0])]
    list_P = [(np.float_(converted_P[i])).tolist() for i in range(odf.shape[0])]
    
    list_N = [len(list_V[i]) for i in range(odf.shape[0])]
    
    [MV5, MV10, MV15, MV20] = [compute_mean(list_V, n) for n in [5, 10, 15, 20]]
    [MP5, MP10, MP15, MP20] = [compute_mean(list_P, n) for n in [5, 10, 15, 20]]
    
    normalized_V10 = [normalize(list_V[i][:10]) for i in range(odf.shape[0])]
    normalized_P10 = [normalize(list_P[i][:10]) for i in range(odf.shape[0])]
    
    [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10] = [single_area(normalized_V10, k) for k in range(10)]
    [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10] = [single_area(normalized_P10, k) for k in range(10)]
    
    dict_df = {'V1':V1, 'V2':V2, 'V3':V3, 'V4':V4, 'V5':V5, 'V6':V6, 'V7':V7, 'V8':V8, 'V9':V9, 'V10':V10,
               'MV5':MV5, 'MV10':MV10, 'MV15':MV15, 'MV20':MV20,
               'P1':P1, 'P2':P2, 'P3':P3, 'P4':P4, 'P5':P5, 'P6':P6, 'P7':P7, 'P8':P8, 'P9':P9, 'P10':P10,
               'MP5':MP5, 'MP10':MP10, 'MP15':MP15, 'MP20':MP20}
    
    return list_N, pd.DataFrame(dict_df)

def models_input(N):
    data0 = create_dataframe(N, observations = 1000)
    data1 = transform_df(data0)
    
    model_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'MV5', 'MV10', 'MV15', 'MV20',
                  'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'MP5', 'MP10', 'MP15', 'MP20']

    X = data1[model_cols].values
    y = data1['type'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=7)
    
    return X_train, X_test, y_train, y_test
```

```{code-cell} ipython3
:tags: [show-output, show-input]

data = pd.read_csv('data_voronoi_cartoradio.csv', sep=',')
list_N, data_test = update_df(data)
print(list_N)
data_test.head()
```

Now that the data are read and transformed, we gather the observations by the number of points $N$ and then create the models inputs for each value of $N$ using the function *models_input(N)*. This task takes so much time to be executed that we got a time-out error on the GitHub server. That's why we prepared locally, for each value of $N$, a file containing the data created locally, and we  read the files directly.


#### N = 31

```{code-cell} ipython3
:tags: [show-output, show-input]

data31 = pd.read_csv('data_31.csv', sep=',')
X31 = data31[model_cols].values
y31 = data31['type'].values
X31_train, X31_test, y31_train, y31_test = train_test_split(X31, y31, test_size=0.3, shuffle=True, random_state=7)
data_test_31 = data_test.iloc[[0, 7]]

print('N = 31')
print('------------------')
print('Logistic Regression Results')
Baseline_LR.fit(X31_train, y31_train)
print(Baseline_LR.predict(data_test_31.values)[0], Baseline_LR.predict_proba(data_test_31.values)[0])
print(Baseline_LR.predict(data_test_31.values)[1], Baseline_LR.predict_proba(data_test_31.values)[1])
print('------------------')
print('Random Forest Results')
baseline_RF.fit(X31_train, y31_train)
print(baseline_RF.predict(data_test_31.values)[0], baseline_RF.predict_proba(data_test_31.values)[0])
print(baseline_RF.predict(data_test_31.values)[1], baseline_RF.predict_proba(data_test_31.values)[1])
```

#### N = 23

```{code-cell} ipython3
:tags: [show-output, show-input]

data23 = pd.read_csv('data_23.csv', sep=',')
X23 = data23[model_cols].values
y23 = data23['type'].values
X23_train, X23_test, y23_train, y23_test = train_test_split(X23, y23, test_size=0.3, shuffle=True, random_state=7)
data_test_23 = data_test.iloc[[1]]

print('N = 23')
print('------------------')
print('Logistic Regression Results')
Baseline_LR.fit(X23_train, y23_train)
print(Baseline_LR.predict(data_test_23.values)[0], Baseline_LR.predict_proba(data_test_23.values)[0])
print('------------------')
print('Random Forest Results')
baseline_RF.fit(X23_train, y23_train)
print(baseline_RF.predict(data_test_23.values)[0], baseline_RF.predict_proba(data_test_23.values)[0])
```

#### N = 22

```{code-cell} ipython3
:tags: [show-output, show-input]

data22 = pd.read_csv('data_22.csv', sep=',')
X22 = data22[model_cols].values
y22 = data22['type'].values
X22_train, X22_test, y22_train, y22_test = train_test_split(X22, y22, test_size=0.3, shuffle=True, random_state=7)
data_test_22 = data_test.iloc[[2, 4, 9]]

print('N = 22')
print('------------------')
print('Logistic Regression Results')
Baseline_LR.fit(X22_train, y22_train)
print(Baseline_LR.predict(data_test_22.values)[0], Baseline_LR.predict_proba(data_test_22.values)[0])
print(Baseline_LR.predict(data_test_22.values)[1], Baseline_LR.predict_proba(data_test_22.values)[1])
print(Baseline_LR.predict(data_test_22.values)[2], Baseline_LR.predict_proba(data_test_22.values)[2])
print('------------------')
print('Random Forest Results')
baseline_RF.fit(X22_train, y22_train)
print(baseline_RF.predict(data_test_22.values)[0], baseline_RF.predict_proba(data_test_22.values)[0])
print(baseline_RF.predict(data_test_22.values)[1], baseline_RF.predict_proba(data_test_22.values)[1])
print(baseline_RF.predict(data_test_22.values)[2], baseline_RF.predict_proba(data_test_22.values)[2])
```

#### N = 24

```{code-cell} ipython3
:tags: [show-output, show-input]

data24 = pd.read_csv('data_24.csv', sep=',')
X24 = data24[model_cols].values
y24 = data24['type'].values
X24_train, X24_test, y24_train, y24_test = train_test_split(X24, y24, test_size=0.3, shuffle=True, random_state=7)
data_test_24 = data_test.iloc[[3, 6]]

print('N = 24')
print('------------------')
print('Logistic Regression Results')
Baseline_LR.fit(X24_train, y24_train)
print(Baseline_LR.predict(data_test_24.values)[0], Baseline_LR.predict_proba(data_test_24.values)[0])
print(Baseline_LR.predict(data_test_24.values)[1], Baseline_LR.predict_proba(data_test_24.values)[1])
print('------------------')
print('Random Forest Results')
baseline_RF.fit(X24_train, y24_train)
print(baseline_RF.predict(data_test_24.values)[0], baseline_RF.predict_proba(data_test_24.values)[0])
print(baseline_RF.predict(data_test_24.values)[1], baseline_RF.predict_proba(data_test_24.values)[1])
```

#### N = 46

```{code-cell} ipython3
:tags: [show-output, show-input]

data46 = pd.read_csv('data_46.csv', sep=',')
X46 = data46[model_cols].values
y46 = data46['type'].values
X46_train, X46_test, y46_train, y46_test = train_test_split(X46, y46, test_size=0.3, shuffle=True, random_state=7)
data_test_46 = data_test.iloc[[5]]

print('N = 46')
print('------------------')
print('Logistic Regression Results')
Baseline_LR.fit(X46_train, y46_train)
print(Baseline_LR.predict(data_test_46.values)[0], Baseline_LR.predict_proba(data_test_46.values)[0])
print('------------------')
print('Random Forest Results')
baseline_RF.fit(X46_train, y46_train)
print(baseline_RF.predict(data_test_46.values)[0], baseline_RF.predict_proba(data_test_46.values)[0])
```

#### N = 29

```{code-cell} ipython3
:tags: [show-output, show-input]

data29 = pd.read_csv('data_29.csv', sep=',')
X29 = data29[model_cols].values
y29 = data29['type'].values
X29_train, X29_test, y29_train, y29_test = train_test_split(X29, y29, test_size=0.3, shuffle=True, random_state=7)
data_test_29 = data_test.iloc[[8]]

print('N = 29')
print('------------------')
print('Logistic Regression Results')
Baseline_LR.fit(X29_train, y29_train)
print(Baseline_LR.predict(data_test_29.values)[0], Baseline_LR.predict_proba(data_test_29.values)[0])
print('------------------')
print('Random Forest Results')
baseline_RF.fit(X29_train, y29_train)
print(baseline_RF.predict(data_test_29.values)[0], baseline_RF.predict_proba(data_test_29.values)[0])
```

We can notice that the classification results are mostly positive, which means that the majority of the samples taken from the CARTORADIO data can be decided as repulsive configurations which is consistent with our starting hypothesis. For the configurations whose results were as non-repulsive, we can say that this is due to one of the two following reasons:
- As long as we are dealing with real data, these two samples may be a non-repulsive ones and the results are actually coherent.
- It is sure that the accuracy of our models is very high, but we may have some classification errors, which means that even if the configuration is repulsive, the model decides that it is not.


## Conclusion

In this paper it has been shown numerically (based on the theoretical results in {cite}`goldman_palm_2010`) that Voronoi cells represent an effective means for determining the nature of repulsion of a configuration (repulsive or not), and this by creating a database of various configurations and extracting the areas and perimeters of the Voronoi cells in order to use them as input to the classification models described earlier.

Once the models are trained and tested on the data created, they are tested after that on real data, which are the positions of a mobile phone base stations in PARIS. Visually, we can easily say that these configurations are repulsive, which we have confirmed for the majority of these configurations by testing them by the previously trained models.

## Bibliography


```{bibliography}
:style: unsrt
```
