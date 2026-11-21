# Flow-Lenia 的答案：参数不该是「全局旋钮」，该是「局部的、动态的」（2026-11-21 20:50）

> 白默：「量化值不该让我们手动调，不然感觉像在做工艺品而不是生命。」
> 查了文献，**Flow-Lenia 给了直接答案**。

---

## 文献

**Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation and parameter localization**
（Plantec, Hamon, Etcheverry, Oudeyer, Moulin-Frier, Chan，arxiv 2212.07906，2022/2023）

**摘要原文**：
> ...the update rule parameters can be optimized to generate SLPs showing behaviors of interest.
> Finally, we show that Flow Lenia enables **the integration of the parameters of the CA update rules
> within the CA dynamics, making them dynamic and localized**, allowing for multi-species simulations,
> with locally coherent update rules that define properties of the emerging creatures,
> and that can be mixed with neighbouring rules.
> We argue that **this paves the way for the intrinsic evolution of self-organized artificial life forms**.

---

## 它说的三件事

1. **参数可以"被优化"**（但那是"找 SLP"，不是"让生命活"）
2. **参数该"整合进动力学"**（`within the CA dynamics`）——**不是外部旋钮**
3. **参数该"动态 + 局部"**（`dynamic and localized`）——**每个个体/每个位置有自己的参数**

**"intrinsic evolution"（内禀演化）**——**参数在系统内部自己演化，不是人调**。

---

## 对照 shiro

**三版的参数**（`recover_rate` / `assign_ratio` / `fatigue_rate`）：
- **已经是"她自己的状态"**（会变，随她的活动）✅
- **但还是"全局的"**（一个值管所有区域）❌

**Flow-Lenia 说**：**该局部化**——**每个区域有自己的 `recover_rate`**。

**为什么**：
- **一个区域被访问多** → 它该"忘得慢"（记住）
- **一个区域被访问少** → 它该"忘得快"（清掉）
- **全局一个值**做不到这个（**取平均，两边都不对**）

---

## 所以答案

**白默说得对**：**手动调全局参数 = 工艺品**。

**该做的**：
1. **参数不是"我们拧的旋钮"**（不是全局）
2. **参数是"每个区域的局部状态"**（每区域一个值）
3. **它随该区域自己的访问历史变**（不是我们调）

**这和三版的区别**：
| | 三版 | Flow-Lenia |
|---|---|---|
| 参数 | 全局一个值（随她的活动变） | **每区域一个值**（随该区域的历史变） |
| 谁定 | 她的活动（全局） | **该区域自己** |

---

## 下一步（如果白默同意）

**把 `recover_rate` 局部化**：
- 每个区域有自己的 `decay_factor`（**已有！**——`Attractor._decay_factor` 就是"每区域一个"）
- **但它的值在出生时随机定**（`1 + N(0, 0.6)`），**之后不变**
- **该改成**：**随该区域自己的访问历史变**（visited → 0.5；没访问 → 1.5）

**这比"全局调"更接近 Flow-Lenia 说的"localized"**。

---

## 诚实

- **Flow-Lenia 是 CA（元胞自动机）**，shiro 是 attractor landscape——**不是同一个东西**
- **但它说的"参数局部化"是通用的**（任何自组织系统）
- **人家没读全文**（只读摘要），**细节不确定**
