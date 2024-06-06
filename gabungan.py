import streamlit as st
import sys
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib.ticker import FuncFormatter

def print_shortest_path(parent, end, vertex_names):
    if parent[end] == -1:
        st.write(vertex_names[end], end=" ")
        return
    print_shortest_path(parent, parent[end], vertex_names)
    st.write(vertex_names[end], end=" ")

def Dijkstra(graph, start, end, vertex_names):
    V = len(graph)
    jarak = [sys.maxsize] * V
    parent = [-1] * V
    visited = [False] * V

    jarak[start] = 0

    for _ in range(V):
        u = min((dist, vertex) for vertex, dist in enumerate(jarak) if not visited[vertex])[1]
        visited[u] = True

        for v, w in enumerate(graph[u]):
            if not visited[v] and w and jarak[u] != sys.maxsize and jarak[u] + w < jarak[v]:
                jarak[v] = jarak[u] + w
                parent[v] = u

    st.write(f"Jalur Terpendek dari {vertex_names[start]} ke {vertex_names[end]}: ", end=" ")
    print_shortest_path(parent, end, vertex_names)
    st.write("\nJarak Terpendek:", jarak[end])

def calculate_distance(graph, path):
    distance = 0
    for i in range(len(path) - 1):
        distance += graph[path[i]][path[i + 1]]
    return distance

def tsp_backtracking(graph, current_position, finish_index, n, count, cost, path, visited, min_cost, best_path):
    if current_position == finish_index:
        if cost < min_cost[0]:
            min_cost[0] = cost
            best_path[0] = path.copy()
        return

    for i in range(n):
        if not visited[i] and graph[current_position][i] > 0:
            visited[i] = True
            path.append(i)
            tsp_backtracking(graph, i, finish_index, n, count + 1, cost + graph[current_position][i], path, visited, min_cost, best_path)
            visited[i] = False
            path.pop()

def solve_tsp(graph, start_index, finish_index):
    n = len(graph)
    visited = [False] * n
    visited[start_index] = True
    path = [start_index]
    min_cost = [sys.maxsize]
    best_path = [[]]

    tsp_backtracking(graph, start_index, finish_index, n, 1, 0, path, visited, min_cost, best_path)
    return best_path[0], min_cost[0]

def format_func(value, tick_number):
    return f"{value:.2f}"

def simulate_execution_times():
    sizes = range(2, 11)
    dijkstra_times = []
    tsp_times = []

    for size in sizes:
        graph = np.random.randint(1, 100, size=(size, size))
        graph = (graph + graph.T) // 2
        start_index = 0
        end_index = size - 1

        start_time = time.time()
        Dijkstra(graph, start_index, end_index, [f"V{i}" for i in range(size)])
        dijkstra_times.append(time.time() - start_time)

        start_time = time.time()
        solve_tsp(graph, start_index, end_index)
        tsp_times.append(time.time() - start_time)

    return sizes, dijkstra_times, tsp_times

def main():
    st.title("Algoritma Jalur Terpendek: Dijkstra dan Backtracking")

    jumlah_vertex = st.number_input("Masukkan jumlah simpul/kota:", min_value=2, step=1, format="%d")

    if jumlah_vertex:
        nama_vertex = []
        for i in range(jumlah_vertex):
            vertex_name = st.text_input(f"Masukkan nama simpul/kota {i+1}:")
            nama_vertex.append(vertex_name)

        graph = [[0 for _ in range(jumlah_vertex)] for _ in range(jumlah_vertex)]
        
        st.write("Masukkan bobot/jarak antar simpul/kota:")
        for i in range(jumlah_vertex):
            bobot_input = st.text_input(f"Masukkan bobot dari {nama_vertex[i]} ke simpul lainnya, pisahkan dengan spasi:", key=f"row_{i}")
            bobot_list = list(map(int, bobot_input.split()))
            if len(bobot_list) != jumlah_vertex:
                st.error(f"Jumlah bobot yang dimasukkan tidak sesuai dengan jumlah simpul.")
                return
            for j in range(jumlah_vertex):
                graph[i][j] = bobot_list[j]

        start_vertex_name = st.text_input("Masukkan simpul/kota awal:")
        end_vertex_name = st.text_input("Masukkan simpul/kota tujuan:")

        if st.button("Hitung Jalur Terpendek"):
            if start_vertex_name in nama_vertex and end_vertex_name in nama_vertex:
                start_index = nama_vertex.index(start_vertex_name)
                end_index = nama_vertex.index(end_vertex_name)
                
                st.write("Menghitung menggunakan Algoritma Dijkstra...")
                start_time = time.time()
                Dijkstra(graph, start_index, end_index, nama_vertex)
                dijkstra_time = time.time() - start_time
                st.write(f"Waktu eksekusi Dijkstra: {dijkstra_time:.6f} detik")

                st.write("Menghitung menggunakan Algoritma Backtracking...")
                start_time = time.time()
                best_path, min_cost = solve_tsp(graph, start_index, end_index)
                tsp_time = time.time() - start_time
                best_path_names = [nama_vertex[i] for i in best_path]
                st.success(f"Rute terpendek: {' -> '.join(best_path_names)}")
                st.success(f"Biaya minimum: {min_cost}")
                st.write(f"Waktu eksekusi Backtracking: {tsp_time:.6f} detik")

                # Plotting
                fig, ax = plt.subplots(figsize=(10, 5))
                algos = ['Dijkstra', 'Backtracking']
                times = [dijkstra_time, tsp_time]
                ax.plot(algos, times, marker='o')
                ax.set_xlabel("Algoritma")
                ax.set_ylabel('Waktu (detik)')
                ax.set_title('Perbandingan Waktu Eksekusi')
                ax.yaxis.set_major_formatter(FuncFormatter(format_func))
                ax.grid(True)
                st.pyplot(fig)

        st.write("Simulasi Kompleksitas Waktu:")
        sizes, dijkstra_times, tsp_times = simulate_execution_times()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sizes, dijkstra_times, label="Dijkstra", marker='o')
        ax.plot(sizes, tsp_times, label="Backtracking", marker='o')
        ax.set_xlabel("Jumlah Simpul")
        ax.set_ylabel('Waktu Eksekusi (detik)')
        ax.set_title('Kompleksitas Waktu Algoritma')
        ax.legend()
        ax.grid(True)
        ax.yaxis.set_major_formatter(FuncFormatter(format_func))
        st.pyplot(fig)

if __name__ == "__main__":
    main()
