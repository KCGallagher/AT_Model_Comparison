# AT_Model_Comparison

Comparison of optimal adaptive therapy schedules under different mathematical models

## Lotka - Volterra Model

This is a simple 2-population Lotka-Volterra tumour model, where $S$ is the number of susceptible cells, and $R$ is the number of resistant cells.

$$
\frac{dS}{dt} = r_{S} S \left(1 - \frac{S+R}{K}\right) \times (1-d_{D}D) - d_{S}S, \\\\
\frac{dR}{dt} = r_{R} R \left(1 - \frac{S+R}{K}\right) - d_{R}R
$$

Both species follow a modified logistic growth model with growth rates $r_{S}$ and $r_{R}$, where the total population (rather than the species population) is modified by the carrying capacity $K$.

For the susceptible population, this growth rate is also modified by the drug concentration $D$ and the killing rate of the drug $d_{D}$.

Finally, both species have a natural death rate, of $d_{S}$ and $d_{R}$ respectively.

This model is implemented in the `LotkaVolterraModel` class, which inherits from `ODEModel`. This parent class sets parameters such as error tolerances for the solver, and then solves the ODE model for each treatment period sequentially.


## Waning Immunity Model

This is a time-varied generalized Lotka-Volterra model (with growth scaling exponent $\alpha$), with separate drug-sensitive $S$ and drug-resistant $R$ cell populations competing for shared resources. However, the resource overlap descreses exponentially, with the resistance index $\gamma > 0$. 

$$
\frac{dS}{dt} = r_{S} S \left[1 - \left(\frac{S + \frac{R}{1 + e^{\gamma t}}}{K_{S}}\right)^{\alpha}  - d_{S}D \right], \\\\
\frac{dR}{dt} = r_{R} R \left[1 - \left(\frac{R + \frac{S}{1 + e^{\gamma t}}}{K_{R}}\right)^{\alpha} - d_{R}D \right]
$$

Each cell species $i$ has a distinct growth rate $r_{i}$, carrying capacity $K_{i}$ and drug-induced death rate $d_{i}$, while logistic growth accounts for net growth. It is worth noting that this model also reduces to the generalized logistic model in the case where $\gamma = 0$ (eliminating the explicitly time-dependent competition), and in the absence of treatment.

This model is implemented in the `ExponentialModel` class.

## Stem-Cell Model

This model distinguishes between prostate cancer stem-like ($S$) and differentiated ($D$) cells to model the tumor response to treatment. Stem-like cells divide at rate $\lambda$, to produce either two stem-like cells (with probability $p_{s}$, but subject to negative feedback $\frac{S}{S+D}$ from differentiated cells), or a stem-like and a non-stem cell. While stem-like cells are androgen-independent and hence do not respond to treatment, differentiated cells die in response to drug application $T_{x}$ at rate $d_{D}$.

$$
    \frac{dS}{dt} =  \left(\frac{S}{S+D}\right) p_{S} \lambda S, \\\\
    \frac{dD}{dt} = \left(1 - \frac{S}{S+D} p_{S}\right) \lambda S - d_{D} T_{x} D
$$

This model is implemented in the `StemCellModel` class.