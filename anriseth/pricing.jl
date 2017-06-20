using Interpolations, Optim
using LaTeXStrings, Plots

function brownsim(T,γ,C,x,amin,amax,Varr,polarr)
    # Dynamics
    # dS_t = -Q(a) dt + σQ(a) dW_t, stopped at zero

    # Solve
    # max \int_0^T aQ(a) dt - CS_T
    #

    Q(a) = 1.-a
    b(a) = -Q(a)
    σ(a) = Q(a)*γ
    f(a) = a*Q(a)
    g(x) = -C*x

    K = length(x)
    Δx = maximum(diff(x))
    sqrtΔx = sqrt(Δx)
    Δt = 0.5*Δx

    # Terminal conditions (reverse time)
    Varr[:,1] .= g(x)

    t = 0.0; ti = 0
    # Start marching
    while t < T - Δt
        ti += 1
        t += Δt
        @show t

        V   = view(Varr,:,ti)
        Vp1 = view(Varr,:,ti+1)
        pol = view(polarr,:,ti+1)
        IV  = interpolate((x,), V, Gridded(Linear())) # Linear extrapolations

        # x = 0 has Dirichlet boundary conditions, x = xmax is just a truncation
        @inbounds Vp1[1] = g(x[1])

        @simd for i=2:K-1
            @inbounds xi = x[i]

            function hjbmin(a)
                y1 = sqrtΔx*σ(a)
                y2 = Δx*b(a)
                x1p = xi + y1
                x1m = xi - y1
                x2  = xi + y2
                # TODO: move V[i] terms outside and assign directly to Vp1[i]
                @inbounds Va = (
                    V[i] + Δt/(2Δx)*(
                        IV[x1p...]-4*V[i]
                        + IV[x1m...]
                        +2*IV[x2...])
                    + Δt*f(a)
                )

                return -Va
            end

            res = optimize(hjbmin,amin,amax)
            @inbounds Vp1[i] = -Optim.minimum(res)
            @inbounds pol[i] = Optim.minimizer(res)
        end # i loop

        # On truncated boundaries, (x[1] ≠ 0)
        # set v(t,x) to be a linear extrapolation of two nearest points in the interior
        @inbounds Vp1[end] = 2*Vp1[end-1]-Vp1[end-2] # x[end] == xmax
    end


end

γ       = 5e-2
C       = 1

xmin    = 0.0
xmax    = 1.0
amin    = 0
amax    = 1.

K       = 51
x       = linspace(xmin,xmax,K)

Δx      = maximum(diff(x))
Δt      = Δx/2

T       = 3.0
tpoints = ceil(Int,T/Δt)+1
tarr    = 0.0:Δt:T+Δt
Varr    = zeros(K,tpoints)
polarr  = zeros(K,tpoints)

println("***Solve HJB equation")
@profile brownsim(T,γ,C,x,amin,amax,Varr, polarr)

Iv = interpolate((x,tarr), Varr, Gridded(Linear()))
Ia = interpolate((x,tarr), polarr, Gridded(Linear()))

pltv = surface(x,tarr, (x,t)-> Iv[x,t],
               xlabel=L"$x$", ylabel=L"$\tau$",
               title=L"Value function $v(\tau,x)$")


plta = surface(x[2:end-1],tarr[2:end-1], (x,t)-> Ia[x,t],
               xlabel=L"$x$", ylabel=L"$\tau$",
               title=L"Optimal price function $a(\tau,x)$")
