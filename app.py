import json
import os
import io
import zipfile
import tempfile  # ←追加
import shutil    # ←追加
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import streamlit as st

# --- 1. データ処理関数 ---

def convert_to_geojson_and_gdf(data):
    """JSONからGeoJSONとGeoDataFrameを作成する"""
    features = []
    items = data if isinstance(data, list) else data.get("features", [])
    if not items and "items" in data:
        items = data["items"]

    for item in items:
        name = "名称不明"
        lat, lon = None, None

        if "geometry" in item and "coordinates" in item["geometry"]:
            coords = item["geometry"]["coordinates"]
            lon, lat = coords[0], coords[1]
            properties = item.get("properties", {})
            name = properties.get("Title") or properties.get("name") or properties.get("title") or "名称不明"
        elif "geometry" in item and "location" in item["geometry"]:
            loc = item["geometry"]["location"]
            lat = loc.get("lat") or loc.get("latitude")
            lon = loc.get("lng") or loc.get("longitude")
            name = item.get("title") or item.get("name") or "名称不明"
        elif "latitude" in item and "longitude" in item:
            lat = item["latitude"]
            lon = item["longitude"]
            name = item.get("name") or item.get("title") or "名称不明"

        if lat is not None and lon is not None:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": {"name": str(name), "description": f"番号/名前: {name}"},
            })

    # GeoDataFrame（空間結合用のデータフレーム）の作成
    if features:
        geometry = [Point(f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1]) for f in features]
        df_patients = pd.DataFrame([f["properties"] for f in features])
        gdf_patients = gpd.GeoDataFrame(df_patients, geometry=geometry, crs="EPSG:4326")
    else:
        gdf_patients = None

    return features, gdf_patients


def process_multiple_hazard_maps(gdf_patients, zip_filepaths, hazard_type):
    """複数のZIPファイルを順番に読み込み、まとめて判定する（クラウド対応版）"""
    all_matched_dfs = []
    
    for zip_path in zip_filepaths:
        if not os.path.exists(zip_path):
            continue
            
        try:
            # 【重要】クラウドのメモリ不足とLinuxの文字化け対策のため、一時フォルダに解凍して処理する
            with tempfile.TemporaryDirectory() as tmpdir:
                
                # 1. ZIPファイルの中身を安全に解凍する
                with zipfile.ZipFile(zip_path, 'r') as z:
                    for info in z.infolist():
                        try:
                            # Windowsの日本語ファイル名をLinuxでも読めるように変換
                            decoded_name = info.filename.encode('cp437').decode('cp932')
                        except Exception:
                            decoded_name = info.filename
                        
                        extracted_path = os.path.join(tmpdir, decoded_name)
                        os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                        
                        # ファイルを書き出す（フォルダの場合はスキップ）
                        if not info.is_dir():
                            with z.open(info) as source, open(extracted_path, "wb") as target:
                                shutil.copyfileobj(source, target)
                
                # 2. 解凍したフォルダの中から対象のShapefileを探す
                target_shps = []
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith('.shp') and '最大' in file:
                            target_shps.append(os.path.join(root, file))
                            
                # 3. 「最大」がなければそれ以外を探す
                if not target_shps:
                    for root, dirs, files in os.walk(tmpdir):
                        for file in files:
                            if file.endswith('.shp') and ('継続' not in file) and ('倒壊' not in file) and ('氾濫' not in file):
                                target_shps.append(os.path.join(root, file))

                if not target_shps:
                    st.warning(f"⚠️ {zip_path} の中に該当する .shp ファイルが見つかりません。")
                    continue

                # 4. 見つかったShapefileをすべて読み込んで判定
                for shp_path in target_shps:
                    try:
                        gdf_hazard = gpd.read_file(shp_path, encoding="cp932")
                    except Exception:
                        gdf_hazard = gpd.read_file(shp_path, encoding="utf-8")

                    if gdf_hazard.crs != "EPSG:4326":
                        gdf_hazard = gdf_hazard.to_crs("EPSG:4326")

                    # 空間結合（判定）
                    joined_data = gpd.sjoin(gdf_patients, gdf_hazard, how="inner", predicate="intersects")

                    # 洪水の 0.5m 以上フィルタリング
                    if hazard_type == "flood" and not joined_data.empty:
                        target_col = None
                        for col_name in ['A31_205', 'A31_105', 'A31_005', 'A31_05', '浸水ランク', '浸水ランクコード']:
                            if col_name in joined_data.columns:
                                target_col = col_name
                                break
                        
                        if not target_col:
                            fallback_cols = [col for col in joined_data.columns if (col.startswith('A31_') and col.endswith('05')) or 'ランク' in col]
                            if fallback_cols:
                                target_col = fallback_cols[0]

                        if target_col:
                            rank_numeric = pd.to_numeric(joined_data[target_col], errors='coerce')
                            danger_codes = [2, 3, 4, 5, 6, 12, 13, 14, 15] 
                            joined_data = joined_data[rank_numeric.isin(danger_codes)]

                    if not joined_data.empty:
                        # 早めに 'name' だけに絞り込んでメモリを節約
                        if 'name' in joined_data.columns:
                            joined_data = joined_data[['name']]
                        all_matched_dfs.append(joined_data)
                    
                    # 1ファイル処理するごとにメモリを強制解放
                    del gdf_hazard
                    
        except Exception as e:
            st.warning(f"⚠️ {zip_path} の読み込み中にエラーが発生しました: {e}")

    # どのファイルでも該当者がいなかった場合
    if not all_matched_dfs:
        return pd.DataFrame()

    # すべての判定結果を1つの表に合体
    final_df = pd.concat(all_matched_dfs, ignore_index=True)
    if 'name' in final_df.columns:
        final_df = final_df.drop_duplicates(subset=['name'])
    
    return final_df


def to_excel_combined(df_tsunami, df_flood, df_landslide):
    """3つの判定結果を1つのExcelファイルの別シートにまとめて出力する"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 津波シート
        if not df_tsunami.empty:
            df_tsunami.to_excel(writer, index=False, sheet_name='津波')
        else:
            pd.DataFrame(columns=['name']).to_excel(writer, index=False, sheet_name='津波')
            
        # 洪水シート
        if not df_flood.empty:
            df_flood.to_excel(writer, index=False, sheet_name='洪水')
        else:
            pd.DataFrame(columns=['name']).to_excel(writer, index=False, sheet_name='洪水')
            
        # 土砂シート
        if not df_landslide.empty:
            df_landslide.to_excel(writer, index=False, sheet_name='土砂')
        else:
            pd.DataFrame(columns=['name']).to_excel(writer, index=False, sheet_name='土砂')
            
    return output.getvalue()


# --- 2. Streamlit UI ---

def main():
    st.set_page_config(page_title="ハザード判定ツール", layout="wide")
    st.title("📍 訪問診療患者 ハザードマップ判定ツール")
    st.write("患者さんの住所(JSON)をアップロードすると、GeoJSONの作成とハザードマップの自動照合を一括で行います。")

    # --- 複数のファイルパスをリスト形式で定義 ---
    DATA_DIR = "data"
    
    tsunami_files = [
        os.path.join(DATA_DIR, "tsunami_i.zip"),
        os.path.join(DATA_DIR, "tsunami_f.zip")
    ]
    flood_files = [
        os.path.join(DATA_DIR, "kouzui_i1.zip"),
        os.path.join(DATA_DIR, "kouzui_i2.zip"),
        os.path.join(DATA_DIR, "kouzui_f1.zip"),
        os.path.join(DATA_DIR, "kouzui_f2.zip")
    ]
    landslide_files = [
        os.path.join(DATA_DIR, "dosha.zip")
    ]

    st.header("患者データのアップロード")
    patient_file = st.file_uploader("GoogleマップのJSONファイル", type=["json"])

    if patient_file is not None:
        # 1. まずGeoJSONへの変換処理を行う
        data = json.load(patient_file)
        features, gdf_patients = convert_to_geojson_and_gdf(data)

        if gdf_patients is None or gdf_patients.empty:
            st.error("位置情報が読み込めませんでした。")
            return

        st.success(f"患者データ {len(features)} 件を読み込みました。ハザード判定を開始します...")

        # 2. ハザード判定処理（実行ボタンを無くし、自動で処理を走らせる）
        with st.spinner("すべてのハザードマップと照合中...（数十秒かかる場合があります）"):
            
            # 津波
            df_tsunami = process_multiple_hazard_maps(gdf_patients, tsunami_files, "tsunami")
            st.write(f"🌊 津波 該当患者: {len(df_tsunami)}件")

            # 洪水
            df_flood = process_multiple_hazard_maps(gdf_patients, flood_files, "flood")
            st.write(f"🌧️ 洪水(0.5m以上) 該当患者: {len(df_flood)}件")

            # 土砂
            df_landslide = process_multiple_hazard_maps(gdf_patients, landslide_files, "landslide")
            st.write(f"⛰️ 土砂災害 該当患者: {len(df_landslide)}件")

        st.success("✅ すべての処理が完了しました！以下のボタンからファイルをダウンロードしてください。")

        # 3. ダウンロード用のデータ準備
        geojson_data = {"type": "FeatureCollection", "features": features}
        geojson_string = json.dumps(geojson_data, ensure_ascii=False, indent=2)
        excel_data = to_excel_combined(df_tsunami, df_flood, df_landslide)

        # 4. ダウンロードボタンを横並びで表示
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="🗺️ 重ねるハザードマップ用 GeoJSON",
                data=geojson_string,
                file_name="hazard_map_points.geojson",
                mime="application/json"
            )
        with col2:
            st.download_button(
                label="📊 判定結果リスト (Excel)",
                data=excel_data,
                file_name="hazard_patients_all.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()