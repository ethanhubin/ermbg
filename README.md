# ERMBG —— 专为游戏资产链路打造的像素级抠图工具

[![Python Version](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-orange?style=flat-square)](https://github.com/ethanhubin/ermbg)

`ERMBG` 是面向游戏 UI、图标、特效和角色资产的自动抠图工具。

## 解决的行业痛点

AI 生成游戏资产时，透明通道（Alpha）仍不稳定。常见问题包括黑白格“假透明”、边缘白边/杂色，以及玻璃、特效等半透明区域破损。这些结果很难直接用于 Unity / Unreal 等游戏引擎。

**ERMBG 的解法是：借鉴影视行业绿幕经验，主动制造纯色背景约束。**
在资产生成阶段，引导 AI 将目标物体生成在纯色（绿幕/蓝幕）背景上。随后由 ERMBG 进行精准的背景扣除与边缘修复，输出像素级干净、可直接投入游戏 UI、动效及角色链路的透明 RGBA PNG。

---

## 设计理念

通用抠图模型（如 Rembg、SAM 等）主要面向真实照片和复杂背景，目标是把主体从背景里分出来。

游戏资产的问题不太一样。UI、图标、特效和角色图经常包含抗锯齿边缘、发光、软阴影和半透明像素。普通模型容易把这些细节当成背景或噪点，结果就是边缘发脏、漏色、阴影断掉，或者透明效果被破坏。

ERMBG 的核心思路是：先识别素材类型，再选择合适的处理路径，最后只修改应该变透明的背景像素，尽量保留素材本身的颜色、边缘和 Alpha 变化。

* **识别游戏资产的边缘细节**：保留抗锯齿、软阴影、发光和粒子半透明，不把它们简单抹掉。
* **按素材类型选择算法**：硬边 UI、玻璃按钮、图标、特效和角色会走不同的 profile 和执行路径。
* **用机制解决问题**：规则基于可观察的图像信号，不围绕单个样本打补丁，方便继续覆盖新的美术素材。

---

## 素材覆盖与智能路由

系统在接收图像后会触发特征识别，自动判断素材类型并分流至对应的处理管线（Pipeline）：

| 素材类型 | 实际游戏场景 | ERMBG 处理优势 | 执行路径 (技术细节) |
| :--- | :--- | :--- | :--- |
| **硬边按钮 / UI 面板** | 扁平化 UI、游戏九宫格框体等 | BG-seed outline trimap 保持硬边、抗锯齿和孔洞 | PyMatting Known-B |
| **玻璃 / 半透明按钮** | 带有透明度梯度与折射的 UI 资产 | 完美保留透明度渐变，杜绝杂色与黑边 | CorridorKey |
| **图标 / Shaped Icon** | 装备图标、技能图标等剪影资产 | 保持图形本身的轮廓结构不畸变 | CorridorKey |
| **特效图标** | 包含 Glow（发光）、烟雾、软 Alpha 边缘 | 完整保留光晕与雾化半透明效果，避免被误判为杂色 | CorridorKey |
| **角色资产** | 2D 角色立绘、带发丝/毛发的怪物资产 | 精准提取发丝级细节与半透明过渡边缘 | CorridorKey |
| **已有 RGBA** | 已包含透明通道的原始图片 | 自动识别并直接放行，避免二次处理损耗 | passthrough |
| **未知 / 不稳定背景** | 未能完美生成在纯色背景上的复杂图像 | 自动切换至兜底策略，最大程度还原边缘 | PyMatting fallback |

---

<!-- ERMBG_EVAL_GALLERY:START -->
## 全量样本结果

来源 batch: `out/direct_worker_game_input_20260610_v002`。后端: `direct-worker`。结果: `88/88` ok。覆盖: button 57 / character 9 / icon 22。

完整展示页见 [docs/eval-gallery.md](docs/eval-gallery.md)。

<details>
<summary>展开 88 个样本缩略图</summary>

<table>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b001" title="B001 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B001.jpg" width="220" alt="B001 result contact sheet"></a><br><sub><b>B001</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b002" title="B002 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B002.jpg" width="220" alt="B002 result contact sheet"></a><br><sub><b>B002</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b003" title="B003 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B003.jpg" width="220" alt="B003 result contact sheet"></a><br><sub><b>B003</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b004" title="B004 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B004.jpg" width="220" alt="B004 result contact sheet"></a><br><sub><b>B004</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b005" title="B005 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B005.jpg" width="220" alt="B005 result contact sheet"></a><br><sub><b>B005</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b006" title="B006 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B006.jpg" width="220" alt="B006 result contact sheet"></a><br><sub><b>B006</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b007" title="B007 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B007.jpg" width="220" alt="B007 result contact sheet"></a><br><sub><b>B007</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b008" title="B008 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B008.jpg" width="220" alt="B008 result contact sheet"></a><br><sub><b>B008</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b009" title="B009 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B009.jpg" width="220" alt="B009 result contact sheet"></a><br><sub><b>B009</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b010" title="B010 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B010.jpg" width="220" alt="B010 result contact sheet"></a><br><sub><b>B010</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b011" title="B011 · button · corridorkey-transparent-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B011.jpg" width="220" alt="B011 result contact sheet"></a><br><sub><b>B011</b> · ok<br>corridorkey-transparent-button<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b012" title="B012 · button · corridorkey-transparent-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B012.jpg" width="220" alt="B012 result contact sheet"></a><br><sub><b>B012</b> · ok<br>corridorkey-transparent-button<br>corridorkey</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b013" title="B013 · button · corridorkey-transparent-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B013.jpg" width="220" alt="B013 result contact sheet"></a><br><sub><b>B013</b> · ok<br>corridorkey-transparent-button<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b014" title="B014 · button · corridorkey-transparent-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B014.jpg" width="220" alt="B014 result contact sheet"></a><br><sub><b>B014</b> · ok<br>corridorkey-transparent-button<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b015" title="B015 · button · corridorkey-transparent-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B015.jpg" width="220" alt="B015 result contact sheet"></a><br><sub><b>B015</b> · ok<br>corridorkey-transparent-button<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b016" title="B016 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B016.jpg" width="220" alt="B016 result contact sheet"></a><br><sub><b>B016</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b017" title="B017 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B017.jpg" width="220" alt="B017 result contact sheet"></a><br><sub><b>B017</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b018" title="B018 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B018.jpg" width="220" alt="B018 result contact sheet"></a><br><sub><b>B018</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b019" title="B019 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B019.jpg" width="220" alt="B019 result contact sheet"></a><br><sub><b>B019</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b020" title="B020 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B020.jpg" width="220" alt="B020 result contact sheet"></a><br><sub><b>B020</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b021" title="B021 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B021.jpg" width="220" alt="B021 result contact sheet"></a><br><sub><b>B021</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b022" title="B022 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B022.jpg" width="220" alt="B022 result contact sheet"></a><br><sub><b>B022</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b023" title="B023 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B023.jpg" width="220" alt="B023 result contact sheet"></a><br><sub><b>B023</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b024" title="B024 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B024.jpg" width="220" alt="B024 result contact sheet"></a><br><sub><b>B024</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b025" title="B025 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B025.jpg" width="220" alt="B025 result contact sheet"></a><br><sub><b>B025</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b026" title="B026 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B026.jpg" width="220" alt="B026 result contact sheet"></a><br><sub><b>B026</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b027" title="B027 · button · corridorkey-transparent-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B027.jpg" width="220" alt="B027 result contact sheet"></a><br><sub><b>B027</b> · ok<br>corridorkey-transparent-button<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b028" title="B028 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B028.jpg" width="220" alt="B028 result contact sheet"></a><br><sub><b>B028</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b029" title="B029 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B029.jpg" width="220" alt="B029 result contact sheet"></a><br><sub><b>B029</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b030" title="B030 · button · corridorkey-transparent-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B030.jpg" width="220" alt="B030 result contact sheet"></a><br><sub><b>B030</b> · ok<br>corridorkey-transparent-button<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b031" title="B031 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B031.jpg" width="220" alt="B031 result contact sheet"></a><br><sub><b>B031</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b032" title="B032 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B032.jpg" width="220" alt="B032 result contact sheet"></a><br><sub><b>B032</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b033" title="B033 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B033.jpg" width="220" alt="B033 result contact sheet"></a><br><sub><b>B033</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b034" title="B034 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B034.jpg" width="220" alt="B034 result contact sheet"></a><br><sub><b>B034</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b035" title="B035 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B035.jpg" width="220" alt="B035 result contact sheet"></a><br><sub><b>B035</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b036" title="B036 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B036.jpg" width="220" alt="B036 result contact sheet"></a><br><sub><b>B036</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b037" title="B037 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B037.jpg" width="220" alt="B037 result contact sheet"></a><br><sub><b>B037</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b038" title="B038 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B038.jpg" width="220" alt="B038 result contact sheet"></a><br><sub><b>B038</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b039" title="B039 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B039.jpg" width="220" alt="B039 result contact sheet"></a><br><sub><b>B039</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b040" title="B040 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B040.jpg" width="220" alt="B040 result contact sheet"></a><br><sub><b>B040</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b041" title="B041 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B041.jpg" width="220" alt="B041 result contact sheet"></a><br><sub><b>B041</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b042" title="B042 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B042.jpg" width="220" alt="B042 result contact sheet"></a><br><sub><b>B042</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b043" title="B043 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B043.jpg" width="220" alt="B043 result contact sheet"></a><br><sub><b>B043</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b044" title="B044 · button · corridorkey-transparent-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B044.jpg" width="220" alt="B044 result contact sheet"></a><br><sub><b>B044</b> · ok<br>corridorkey-transparent-button<br>corridorkey</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b045" title="B045 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B045.jpg" width="220" alt="B045 result contact sheet"></a><br><sub><b>B045</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b046" title="B046 · button · corridorkey-transparent-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B046.jpg" width="220" alt="B046 result contact sheet"></a><br><sub><b>B046</b> · ok<br>corridorkey-transparent-button<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b047" title="B047 · button · corridorkey-transparent-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B047.jpg" width="220" alt="B047 result contact sheet"></a><br><sub><b>B047</b> · ok<br>corridorkey-transparent-button<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b048" title="B048 · button · corridorkey-transparent-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B048.jpg" width="220" alt="B048 result contact sheet"></a><br><sub><b>B048</b> · ok<br>corridorkey-transparent-button<br>corridorkey</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b049" title="B049 · button · corridorkey-transparent-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B049.jpg" width="220" alt="B049 result contact sheet"></a><br><sub><b>B049</b> · ok<br>corridorkey-transparent-button<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b050" title="B050 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B050.jpg" width="220" alt="B050 result contact sheet"></a><br><sub><b>B050</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b051" title="B051 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B051.jpg" width="220" alt="B051 result contact sheet"></a><br><sub><b>B051</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b052" title="B052 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B052.jpg" width="220" alt="B052 result contact sheet"></a><br><sub><b>B052</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b053" title="B053 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B053.jpg" width="220" alt="B053 result contact sheet"></a><br><sub><b>B053</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b054" title="B054 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B054.jpg" width="220" alt="B054 result contact sheet"></a><br><sub><b>B054</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b055" title="B055 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B055.jpg" width="220" alt="B055 result contact sheet"></a><br><sub><b>B055</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b056" title="B056 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B056.jpg" width="220" alt="B056 result contact sheet"></a><br><sub><b>B056</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#b057" title="B057 · button · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/B057.jpg" width="220" alt="B057 result contact sheet"></a><br><sub><b>B057</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i001" title="I001 · icon · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I001.jpg" width="220" alt="I001 result contact sheet"></a><br><sub><b>I001</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i002" title="I002 · icon · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I002.jpg" width="220" alt="I002 result contact sheet"></a><br><sub><b>I002</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i003" title="I003 · icon · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I003.jpg" width="220" alt="I003 result contact sheet"></a><br><sub><b>I003</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i004" title="I004 · icon · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I004.jpg" width="220" alt="I004 result contact sheet"></a><br><sub><b>I004</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i005" title="I005 · icon · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I005.jpg" width="220" alt="I005 result contact sheet"></a><br><sub><b>I005</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i006" title="I006 · icon · corridorkey-shaped-icon"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I006.jpg" width="220" alt="I006 result contact sheet"></a><br><sub><b>I006</b> · ok<br>corridorkey-shaped-icon<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i007" title="I007 · icon · corridorkey-shaped-icon"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I007.jpg" width="220" alt="I007 result contact sheet"></a><br><sub><b>I007</b> · ok<br>corridorkey-shaped-icon<br>corridorkey</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i008" title="I008 · icon · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I008.jpg" width="220" alt="I008 result contact sheet"></a><br><sub><b>I008</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i009" title="I009 · icon · pymatting-hard-button"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I009.jpg" width="220" alt="I009 result contact sheet"></a><br><sub><b>I009</b> · ok<br>pymatting-hard-button<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i010" title="I010 · icon · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I010.jpg" width="220" alt="I010 result contact sheet"></a><br><sub><b>I010</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i011" title="I011 · icon · known-bg-glow"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I011.jpg" width="220" alt="I011 result contact sheet"></a><br><sub><b>I011</b> · ok<br>known-bg-glow<br>known_bg_glow</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i012" title="I012 · icon · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I012.jpg" width="220" alt="I012 result contact sheet"></a><br><sub><b>I012</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i013" title="I013 · icon · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I013.jpg" width="220" alt="I013 result contact sheet"></a><br><sub><b>I013</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i014" title="I014 · icon · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I014.jpg" width="220" alt="I014 result contact sheet"></a><br><sub><b>I014</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i015" title="I015 · icon · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I015.jpg" width="220" alt="I015 result contact sheet"></a><br><sub><b>I015</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i016" title="I016 · icon · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I016.jpg" width="220" alt="I016 result contact sheet"></a><br><sub><b>I016</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i017" title="I017 · icon · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I017.jpg" width="220" alt="I017 result contact sheet"></a><br><sub><b>I017</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i018" title="I018 · icon · known-bg-glow"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I018.jpg" width="220" alt="I018 result contact sheet"></a><br><sub><b>I018</b> · ok<br>known-bg-glow<br>known_bg_glow</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i019" title="I019 · icon · known-bg-glow"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I019.jpg" width="220" alt="I019 result contact sheet"></a><br><sub><b>I019</b> · ok<br>known-bg-glow<br>known_bg_glow</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i020" title="I020 · icon · known-bg-glow"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I020.jpg" width="220" alt="I020 result contact sheet"></a><br><sub><b>I020</b> · ok<br>known-bg-glow<br>known_bg_glow</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i021" title="I021 · icon · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I021.jpg" width="220" alt="I021 result contact sheet"></a><br><sub><b>I021</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#i022" title="I022 · icon · pymatting-known-bg"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/I022.jpg" width="220" alt="I022 result contact sheet"></a><br><sub><b>I022</b> · ok<br>pymatting-known-bg<br>pymatting_known_b</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#c001" title="C001 · character · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/C001.jpg" width="220" alt="C001 result contact sheet"></a><br><sub><b>C001</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#c002" title="C002 · character · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/C002.jpg" width="220" alt="C002 result contact sheet"></a><br><sub><b>C002</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#c003" title="C003 · character · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/C003.jpg" width="220" alt="C003 result contact sheet"></a><br><sub><b>C003</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#c004" title="C004 · character · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/C004.jpg" width="220" alt="C004 result contact sheet"></a><br><sub><b>C004</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#c005" title="C005 · character · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/C005.jpg" width="220" alt="C005 result contact sheet"></a><br><sub><b>C005</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
</tr>
<tr>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#c006" title="C006 · character · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/C006.jpg" width="220" alt="C006 result contact sheet"></a><br><sub><b>C006</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#c007" title="C007 · character · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/C007.jpg" width="220" alt="C007 result contact sheet"></a><br><sub><b>C007</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#c008" title="C008 · character · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/C008.jpg" width="220" alt="C008 result contact sheet"></a><br><sub><b>C008</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
<td width="25%" valign="top"><a href="docs/eval-gallery.md#c009" title="C009 · character · corridorkey-character"><img src="docs/assets/eval-gallery/direct_worker_game_input_20260610_v002/C009.jpg" width="220" alt="C009 result contact sheet"></a><br><sub><b>C009</b> · ok<br>corridorkey-character<br>corridorkey</sub></td>
</tr>
</table>

</details>
<!-- ERMBG_EVAL_GALLERY:END -->

---

## 真实执行主线

```text
input
  -> Preprocess
  -> Analyze
  -> Decide
  -> Execute
  -> Output
```

- Preprocess 处理棋盘格/背景场归一化等输入问题。
- Analyze 生成 route candidates、semantic candidates 和轻量 preview。
- Decide 选择默认或用户指定候选。
- Execute 只运行一次最终 request。

Known-B 当前主线由 Analyze 生成 explicit trimap：从强置信 BG seed 往内搜索到真实
outline，填充 outline 内部作为 FG core，边缘/过渡/shadow-facing 区域作为 unknown。
孔洞作为候选 overlay 到 trimap；shadow 不再作为独立语义候选。

## 自动化路由机制 (Execution Profile)

图片特征识别在执行前自动确定配置（Profile）：


```

[ 输入图像 ] ──> ( 特征自动识别 ) ──> 决定 [ Execution Profile ] ──> 路由至最强 [ 执行路径 ]

```

| 素材 / 场景 | Execution Profile | 执行路径 |
| :--- | :--- | :--- |
| clean RGBA | `passthrough` | passthrough |
| 硬边 UI / 确定性按钮 | `pymatting-hard-button` | PyMatting Known-B |
| 已知背景 fallback | `pymatting-known-bg` | PyMatting Known-B |
| 未知 / 不稳定背景 | `pymatting-fallback` | PyMatting fallback |
| shaped icon | `corridorkey-shaped-icon` | CorridorKey |
| effect icon | `corridorkey-effect-icon` | CorridorKey |
| 半透明 / 玻璃按钮 | `corridorkey-transparent-button` | CorridorKey |
| 角色 | `corridorkey-character` | CorridorKey |

---

## 📤 输出字段说明

每次抠图任务均会输出丰富的数据结构，方便下游工具链无缝承接：

| 字段 | 说明 |
| :--- | :--- |
| `rgba` | 可直接使用的 RGBA PNG |
| `alpha` | float32 `[0, 1]` soft mask |
| `foreground_srgb` | sRGB foreground companion |
| `strategy_name` | 实际执行策略 |
| `background_color` | 诊断到的背景色 |
| `debug.auto_route` | 识别结果、asset kind、profile 和 backend 选择 |
| `server_elapsed_sec` | 服务端耗时 |

---

## 📦 安装指南

需要 Python 3.12。 建议使用 `uv` 进行虚拟环境管理与依赖安装：

```bash
uv venv .venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -e ".[web,dev,torch]"

```

> 💡 `torch` extra 用于 Direct Worker 的 CorridorKey 路径。

---

## 🚀 使用方法

### 1. Web UI 界面

```powershell
.\scripts\start_local.ps1

```

Web 默认选 `Auto Route`。手动下拉只选 algorithm（`CorridorKey`、`PyMatting Known-B`、`Known-B Glow`、`Passthrough`）。

### 2. CLI 命令行

```bash
.venv/bin/ermbg matte input.png --backend auto --out-dir out/result

```

`--backend` 可选值：`auto`、`pymatting-known-b`、`corridorkey`、`known_bg_glow`、`passthrough`。

### 3. Python API 接入

```python
from pathlib import Path
from PIL import Image
from ermbg.api import matte_image
from ermbg.io import save_rgba, save_mask

result = matte_image(
    Image.open("input.png").convert("RGBA"),
    backend="auto",
    output_dir=Path("out/result"),
)
save_rgba("out/result/output.png", result.rgba)
save_mask("out/result/alpha.png", result.alpha)
print(result.debug)

```

> 💡 重要 metadata 字段：`execution_profile`、`parameter_profile`、`debug.auto_route.algorithm`、`server_elapsed_sec`。Direct Worker 还会返回 `debug.direct_worker.execution_backend`（如 `direct-corridorkey`、`direct-pymatting-known-b`）。

### 4. Game Eval (批量回归验证)

批量回归验证，输出写入 `out/` 下的 batch 目录并生成 `summary.json`。

```bash
.venv/bin/python scripts/run_corridorkey_game_eval.py \
  --backend auto \
  --sample-id B001,I011,C001 \
  --out-dir out/smoke

```

去掉 `--sample-id` 跑完整 88 样本集。回归样本集：`samples/corridorkey_semantic/manifest.json`。

---

## 🌐 部署与分布式架构

配置写在 `ermbg.config.json`（共享默认值）和 gitignored `ermbg.local.json`（机器相关覆盖）。`services.direct_worker_urls` 是 Direct Worker URL 优先级列表，支持自动 fallback。

### 远端 Direct Worker 部署：

```bash
# 先同步当前源码快照，再重启远端 Direct Worker
scripts/sync_comfy_ssh.sh --clean --smoke
scripts/restart_direct_worker_ssh.sh --restart
curl -sS "http://192.168.0.8:7871/health"

```

### 本机前端 Web + 远端 Worker 联动：

```powershell
$env:ERMBG_DIRECT_URL = "<services.direct_worker_url>"
.\scripts\start_local.ps1 -SkipDirectWorker -DirectUrl $env:ERMBG_DIRECT_URL

```

详见 [docs/modules/operations.md](docs/modules/operations.md)。

---

## 🧪 自动化测试与验证

```bash
# 单元测试
.venv/bin/pytest -q

# Direct Worker HTTP smoke
.venv/bin/python scripts/smoke_direct_worker_http.py \
  --base-url <services.direct_worker_url> \
  --sample-id B001,I011

# Runtime capabilities
curl -sS "<web-url>/api/runtime-capabilities"

```

---

## 🗺️ 项目地图 (Project Map)

| 路径 | 角色 |
| --- | --- |
| `ermbg/router.py` | route 决策、asset kind、execution profile |
| `ermbg/api.py` | 主 matting API 和 PyMatting Known-B 实现 |
| `ermbg/analyze.py` | Analyze、route/semantic candidates、Known-B explicit trimap preview |
| `ermbg/pymatting_refine.py` | Known-B BG-seed outline trimap builder |
| `ermbg/corridorkey_runner.py` | 进程内 CorridorKey runner |
| `ermbg/direct_worker.py` | direct 执行编排 |
| `ermbg/direct_worker_client.py` | Direct Worker HTTP client |
| `ermbg/direct_worker_server.py` | 远端 Direct Worker FastAPI 服务 |
| `ermbg/web.py` | Web UI、Web API、Game Eval |
| `scripts/run_corridorkey_game_eval.py` | 批量 eval |
| `samples/corridorkey_semantic/` | B/I/C 游戏素材样本集 |
| `out/` | eval batch、summary、debug 产物 |

---

## 📄 参考文档

* [docs/README.md](docs/README.md) — 文档入口和阅读顺序
* [docs/architecture.md](docs/architecture.md) — 主线架构与服务边界
* [docs/modules/route-profiles.md](docs/modules/route-profiles.md) — route / profile / backend 契约
* [docs/modules/operations.md](docs/modules/operations.md) — 完整安装与启动流程
