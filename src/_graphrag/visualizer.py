import os
import networkx as nx
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import japanize_matplotlib

def visualize_graph(nodes_file, relationships_file, output_file):
    # フォントの設定
    plt.rcParams['font.family'] = 'IPAexGothic'
    
    # ノードとエッジのデータを読み込む
    nodes_df = pq.read_table(nodes_file).to_pandas()
    relationships_df = pq.read_table(relationships_file).to_pandas()

    # グラフを作成
    G = nx.Graph()

    # ノードを追加
    for _, node in nodes_df.iterrows():
        G.add_node(node['title'], type=node['type'])

    # エッジを追加
    for _, rel in relationships_df.iterrows():
        G.add_edge(rel['source'], rel['target'], weight=rel['weight'])

    # レイアウトを設定
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # グラフを描画
    plt.figure(figsize=(20, 16))
    
    # ノードの色をタイプに基づいて設定
    color_map = {'PERSON': '#FFA07A', 'GEO': '#98FB98', 'EVENT': '#87CEFA'}
    node_colors = [color_map.get(G.nodes[node]['type'], '#DCDCDC') for node in G.nodes()]

    # ノードの大きさを接続数に基づいて設定
    node_sizes = [300 * (1 + G.degree(node)) for node in G.nodes()]

    # エッジの太さを重みに基づいて設定
    edge_widths = [rel['weight'] * 0.5 for (_, _, rel) in G.edges(data=True)]

    # グラフを描画
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=node_sizes, font_size=8, font_weight='bold', font_family='IPAexGothic',
            edge_color='#CCCCCC', width=edge_widths, alpha=0.7)
    
    # エッジの重みをラベルとして表示
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # タイトルを追加
    plt.title("エンティティ関係グラフ", fontsize=20)

    # 凡例を追加
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'{k}',
                                  markerfacecolor=v, markersize=10)
                       for k, v in color_map.items()]
    plt.legend(handles=legend_elements, loc='upper right')

    # グラフを保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# 使用例
if __name__ == "__main__":
    visualize_graph(
        os.path.join(os.environ['RAG_ROOT'], 'db/graphrag/create_final_nodes.parquet'),
        os.path.join(os.environ['RAG_ROOT'], 'db/graphrag/create_final_relationships.parquet'),
        os.path.join(os.environ['RAG_ROOT'], 'figures', 'graph_visualization.png')
    )