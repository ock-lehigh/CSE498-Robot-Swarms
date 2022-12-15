import numpy as np
from math import *
import matplotlib.pyplot as plt

# np.random.seed(1)

N = 10

E1 = np.mat([[0, 1], [1, 0]])
E2 = np.mat([[1, 0], [0, -1]])
E3 = np.mat([[0, -1], [1, 0]])
I = np.mat([[1, 0], [0, 1]])


q = np.mat(np.random.random(size=(N, 2)))


def calculate_μ(q, N):
    μ = np.mat(np.zeros(2))
    for i in range(N):
        μ = μ + q[i]
    return μ / N

def calculate_θ(q, μ, N):
    tx = 0
    ty = 0
    for i in range(N):
        tx = tx + (q[i] - μ) * E1 * np.transpose(q[i] - μ)
        ty = ty + (q[i] - μ) * E2 * np.transpose(q[i] - μ)
    return 0.5 * atan2(tx, ty)

def calculate_s1_s2(q, μ, N, H1, H2):
    s1 = 0
    s2 = 0
    for i in range(N):
        s1 = s1 + (q[i] - μ) * H1 * np.transpose(q[i] - μ)
        s2 = s2 + (q[i] - μ) * H2 * np.transpose(q[i] - μ)
    return s1/(2*(N-1)), s2/(2*(N-1))

def update(q, N):
    μ = calculate_μ(q, N)
    θ = calculate_θ(q, μ, N)

    R = np.mat([[cos(θ), -sin(θ)] ,[sin(θ), cos(θ)]])

    H1 = I + R * R * E2
    H2 = I - R * R * E2
    H3 = R * R * E1  

    s1, s2 = calculate_s1_s2(q, μ, N, H1, H2)

    return μ, θ, R, H1, H2, H3, s1, s2


μ, θ, R, H1, H2, H3, s1, s2 = update(q, N)


def generate_plot(q, N, μ, R, s1, s2):
    a = sqrt((N-1)*s1)
    b = sqrt((N-1)*s2)

    rect_list = []
    offset = R * np.transpose(np.mat([a,b]))
    rect_list.append([μ[0,0]+offset[0,0], μ[0,1]+offset[1,0]])
    offset = R * np.transpose(np.mat([a,-b]))
    rect_list.append([μ[0,0]+offset[0,0], μ[0,1]+offset[1,0]])
    offset = R * np.transpose(np.mat([-a,-b]))
    rect_list.append([μ[0,0]+offset[0,0], μ[0,1]+offset[1,0]])
    offset = R * np.transpose(np.mat([-a,b]))
    rect_list.append([μ[0,0]+offset[0,0], μ[0,1]+offset[1,0]])
    offset = R * np.transpose(np.mat([a,b]))
    rect_list.append([μ[0,0]+offset[0,0], μ[0,1]+offset[1,0]])

    rect = np.array(rect_list)

    xs, ys = zip(*rect)

    plt.clf()
    plt.xlim(-1,8)
    plt.ylim(-1,8)
    plt.grid(True)
    plt.plot(xs,ys)
    plot_obstacles()
    for i in range(N):
        plt.plot(q[i,0], q[i,1], 'bo')
    plt.pause(0.01)

def plot_obstacles():
    plt.plot(3, 3.3, 'ro')
    plt.plot(5.5, 5, 'ro')

def obstacle_avoidence(N, μ, s1, s2, p):
    a = sqrt((N-1)*s1)
    b = sqrt((N-1)*s2)

    distance = sqrt(pow(p[0]-μ[0,0],2)+pow(p[1]-μ[0,1],2)) - sqrt(pow(a,2)+pow(b,2))
    return [0.1*(p[0]-μ[0,0])/pow(distance,3), 0.1*(p[1]-μ[0,1])/pow(distance,3)]

μd = np.mat([[7,7]])
θd = 0
s1d = 0.1
s2d = 0.1

kμ = np.mat([[0.5, 0], [0, 0.5]])
kθ = 1
ks1 = 0.2
ks2 = 0.2

tf = 10.
Δt = 0.05
time = np.linspace(0., tf, int(tf / Δt) + 1)

plt.figure(figsize=(12,12))
plt.ion()


for t in time:
    uμ = (μd - μ) * kμ
    uθ = kθ * (θd - θ)
    us1 = ks1 * (s1d - s1)
    us2 = ks2 * (s2d - s2)
    uq = np.mat(np.random.random(size=(N, 2)))

    for i in range(N):
        uq[i] = uμ + np.dot(uθ*(s1-s2)/(s1+s2), np.transpose(H3*np.transpose((q[i]-μ)))) \
         + np.dot(us1/(4*s1), np.transpose(H1*np.transpose(q[i]-μ))) + np.dot(us2/(4*s2), np.transpose(H2*np.transpose(q[i]-μ)))
        p = [3, 3.3]
        uq[i] -= obstacle_avoidence(N, μ, s1, s2, p)
        p = [5.5, 5]
        uq[i] -= obstacle_avoidence(N, μ, s1, s2, p)
        q[i] += uq[i]*Δt

    μ, θ, R, H1, H2, H3, s1, s2 = update(q, N)

    generate_plot(q, N, μ, R, s1, s2)


plt.ioff()
plt.show()
