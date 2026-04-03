---
layout: post
title: "Analyzing Qwen2.5-VL"
date: 2026-04-03 09:00:00 +0800
excerpt: "Qwen2.5-VL stands out for four practical changes: window attention in the vision encoder, dynamic FPS sampling for video, temporal MRoPE aligned to absolute time, and stronger multimodal data curation."
summary: "Qwen2.5-VL stands out for four practical changes: window attention in the vision encoder, dynamic FPS sampling for video, temporal MRoPE aligned to absolute time, and stronger multimodal data curation."
categories:
  - analysis
tags:
  - qwen2.5-vl
  - vision-language
permalink: /posts/analyzing-qwen2-5-vl/
---
Qwen2.5-VL follows the standard Encoder -> Merger -> Decoder architecture, with a few meaningful changes in both architecture and training. This post is a technical note on the new contributions and the inference pipeline.

There were mainly 4 contributions made:
(1) implement **window attention** in the visual encoder to optimize inference efficiency; 
(2) introduce **dynamic FPS sampling**, extending dynamic resolution to the temporal 
dimension and enabling comprehensive video understanding across varied sampling rates;
(3) upgrade **MRoPE in the temporal domain** by aligning to absolute time, thereby facilitating more sophisticated temporal sequence learning
(4)  significant efforts in curating high-quality data for both pre-training and supervised fine-tuning

![Overview of the four main Qwen2.5-VL contributions]({{ '/assets/images/qwen2-5-vl/encoder-overview.png' | relative_url }})

## The encoder

Converts raw images or video frames into tokens that are compact enough for the LLM to process, while still preserving fine spatial structure, scale, and temporal cues. Qwen2.5-VL tackles this with a redesigned ViT that combines **patchification**, **2D rotary positional encoding**, and **mostly local window attention with a few global-attention layers**. In the paper’s released configuration, the visual encoder has **32 layers**, uses **14×14 patches**, sets a **window size of 112×112 pixels**, and keeps only **4 full-attention blocks** at layers **7, 15, 23, and 31**.

Before getting into Qwen-specific choices, it helps to separate the encoder’s problem into two parts. First, the model needs to know **where** a patch came from. A transformer, by itself, is permutation-equivariant: if you shuffle tokens, self-attention does not inherently know they were reordered. Second, the model needs to decide **which other patches each patch should communicate with**. That is the attention pattern question. RoPE addresses the first problem; window attention addresses the second.

### From pixels to patches

Qwen2.5-VL begins like a Vision Transformer. Images are resized so height and width are multiples of **28**, then partitioned into patches with stride **14**. Each patch becomes one visual token before later compression. For video, the encoder extends patchification into the temporal dimension by grouping **two consecutive frames** together, reducing token count without abandoning temporal structure. This is one of the reasons the model can afford longer videos than a naïve “encode every frame independently” pipeline.

Turning height and width into the multiples of 28 enables the model to accept the images and videos at (almost) their native resolution, given the example in the image the height = 8204 and width =1092, we have $8204/28 = 293$, $1092/28 = 39$, then $293 * 39 = 11427$ tokens. Instead of setting a fixed resolution for the encoder, this way supports the native resolution to be processed. 

PS: Sometimes the resolution might not come in a multiple of 28, and resizing is needed, and that's why it is (almost) native resolution, but it still keeps the aspect ratio roughly the same.

![Example showing how native-resolution patching preserves a tall document layout]({{ '/assets/images/qwen2-5-vl/native-resolution-example.png' | relative_url }})

### What is RoPE?

**Rotary Position Embedding (RoPE)** is a way to inject positional information directly into the query and key vectors used by attention. Instead of adding a learned position vector to each token embedding, RoPE rotates each 2D pair of hidden dimensions by a position-dependent angle. The original RoFormer paper’s key point is that RoPE uses an **absolute position parameterization**, but the resulting attention score becomes a function of **relative position difference**. That is the elegant part: positions are encoded locally in each token, yet the dot product naturally reflects offsets between tokens.

In a 1D sequence, for token position *m*, each 2D subspace of the query or key is rotated by an angle proportional to *m*:

$$
R(m\theta_i)=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
$$

Then the rotated query and key are:

$$
q_m' = R(m\theta)\, q_m
$$

$$
k_n' = R(n\theta)\, k_n
$$

The important consequence is:

$$
{q_m'}^\top k_n' = q_m^\top R((n-m)\theta)\, k_n
$$

So although each vector was rotated using its own absolute position, the attention interaction depends on the **difference** $n-m$. That is why RoPE often feels like “absolute encoding with relative behavior.” The RoFormer paper also emphasizes two other benefits: it extrapolates to longer sequences better than many absolute embeddings, and it biases interactions to decay as relative distance increases.

Intuitively, RoPE does not attach a sticky “position label” to a token. It changes the **coordinate system** in which that token is represented. Two nearby tokens are rotated similarly, so they remain easy to align; two far-apart tokens are rotated more differently, so their interaction changes accordingly.

### Why 1D RoPE is not enough for images

A text sequence is fundamentally one-dimensional. An image is not. If you flatten image patches into a single sequence, token 17 being next to token 18 in the flattened order does not necessarily mean the same thing as “one pixel neighborhood away” in the original image. Flattening destroys the symmetry of the 2D plane unless the positional encoding repairs it.

That is exactly why Qwen2-VL replaced absolute position embeddings in the ViT with **2D-RoPE**. The Qwen2-VL paper states that the ViT removes the original absolute embeddings and introduces **2D-RoPE** to capture two-dimensional positional information, enabling native dynamic resolution processing. The broader ECCV 2024 study on RoPE for Vision Transformers likewise frames 2D RoPE as a 1D-to-2D expansion of RoPE for images, and shows it is especially beneficial when resolution changes at inference time.

### What is 2D RoPE?

The simplest way to think about **2D RoPE** is: instead of one position index *m*, each patch has two coordinates, $(x,y)$. The hidden channels are split so that part of the rotation tracks horizontal position and another part tracks vertical position. In the common axial form, one set of dimensions rotates according to $x$, another according to $y$. Then the attention score between patch $(x_1,y_1)$ and patch $(x_2,y_2)$ depends on the relative offset $(x_2-x_1,\; y_2-y_1)$, not just a flattened 1D gap.

A compact expression is:

$$
q_{x,y}' = R_x(x)\, R_y(y)\, q_{x,y}
$$

$$
k_{u,v}' = R_x(u)\, R_y(v)\, k_{u,v}
$$

So the dot product becomes sensitive to both horizontal and vertical displacement. That is much closer to how image geometry actually works. A patch shifted right by 3 and down by 2 should not look identical to one shifted right by 5 and down by 0, even if both happen to be the same flattened sequence distance away.

There is one subtlety worth noticing. The 2D RoPE literature distinguishes between **axial** variants, which treat horizontal and vertical axes separately, and more expressive variants that try to better capture diagonal structure. The ECCV 2024 paper argues that pure axial 2D RoPE is weaker for diagonal directions and proposes a mixed-axis alternative. Qwen’s papers do not present their encoder as introducing a new general-purpose 2D RoPE variant for vision; rather, they use 2D RoPE as part of a practical native-resolution ViT design.

### From 2D RoPE to Qwen’s multimodal version

Qwen2-VL goes one step further with **M-RoPE**. Instead of only height and width, it decomposes position into **temporal, height, and width** components. For text, those three IDs collapse to the same 1D behavior. For images, temporal ID stays constant while height and width vary by patch location. For video, temporal IDs increase across frames while height and width continue to encode spatial location. Qwen2.5-VL then improves the temporal component further by aligning it to **absolute time**, not just frame count, so the spacing between temporal IDs reflects the pace of the video under different FPS sampling schedules.

That broader multimodal story matters because the encoder is not just an image encoder anymore. It is a spatiotemporal encoder that must keep geometry and timing coherent before handing tokens to the LLM.

### Global self-attention

In a standard ViT block with **global self-attention**, every patch attends to every other patch. If the image produces $N$ patch tokens, attention constructs an $N \times N$ interaction pattern. This is conceptually beautiful: every patch can immediately compare itself against the entire image. But it is also computationally expensive because the cost scales quadratically:

$$
O(N^2)
$$

As image resolution rises, $N$ rises, and the attention matrix grows much faster than linearly. That is exactly the bottleneck Qwen2.5-VL points to when discussing native-resolution inputs.

For example, if an image doubles its height and width in patches, the number of tokens grows by roughly $4\times$, but global attention cost grows by roughly $16\times$. That is why “just keep full attention everywhere” quickly becomes impractical for high-resolution documents, long charts, or long videos.

### Contribution 1: Window attention: local neighborhoods instead of a full social network

**Window attention** limits self-attention to small local regions. Instead of every patch attending to the whole image, a patch only attends to patches inside its own window. This idea is closely associated with Swin Transformer, whose central observation is that local windows give much better efficiency, while shifted windows can still allow cross-window communication. Swin explicitly describes the computational complexity as **linear in image size** for fixed window size.

Mathematically, suppose the image has $N$ tokens and each attention window contains $M^2$ tokens. Then there are about $N / M^2$ windows. Each window performs attention over only $M^2$ tokens, costing $O(M^4)$ per window, so the total cost is:

$$
O\!\left(\frac{N}{M^2} \cdot M^4\right) = O(NM^2)
$$

For fixed $M$, that scales linearly with $N$, not quadratically. This is the core computational win.

The trade-off is obvious. Global attention is expensive but sees everything immediately. Window attention is cheap but sees only local neighborhoods in one layer. So a model using pure window attention everywhere can become myopic unless it has some other mechanism for long-range mixing.

Also I would personally think that window attention introduces the locality bias like CNN, window attention introduces a CNN-like **locality prior**, but not a full CNN-style inductive bias. CNNs hard-code both local receptive fields and shared filters, whereas window attention hard-codes locality but keeps the actual interactions content-dependent through self-attention.

### How Qwen2.5-VL balances locality and global context

Qwen2.5-VL’s encoder does not choose between global and local attention once and for all. It mixes them. Most layers use **windowed attention**, but **four layers** use **full self-attention**. In the released configuration, those full-attention blocks are at layers **7, 15, 23, and 31**. The paper does not frame this in exactly these words, but a natural interpretation is that the model alternates between cheap local processing and periodic global synchronization. Local windows efficiently capture nearby texture, layout, and object details; occasional full-attention layers let information travel across the whole image.

This design is especially sensible for documents and high-resolution scenes. Many dependencies are local: letters form words, words form lines, lines form tables. But some dependencies are global: the title relates to the footer, a legend explains a chart across the page, or an answer depends on matching objects in distant regions. Local-only attention would struggle with the latter; global-only attention would pay too much compute for the former.

### What “window size 112” actually means

Qwen2.5-VL reports a **window size of 112×112**, with **patch size 14**. That means each attention window spans **8×8 patches**, since:

$$
\frac{112}{14} = 8
$$

So one local attention region covers **64 patch tokens** before the next layer. The paper also notes that regions smaller than 112×112 are processed **without padding**, which preserves the native resolution of smaller regions rather than distorting them to fit an arbitrary fixed-size window.

This is a quiet but important engineering detail. Native-resolution support is not just about letting very large images through. It is also about not gratuitously warping small or oddly shaped inputs into a square template. For OCR, charts, UI screenshots, and long documents, preserving original geometry matters.

### Why 2D RoPE and window attention belong together

2D RoPE and window attention solve different problems, but they reinforce each other.

A local attention window without good positional encoding can still confuse relative layout. It knows which tokens are in the same neighborhood, but not necessarily the exact 2D relationship between them. Conversely, a strong 2D positional encoding without an efficient attention pattern still leaves the model with a quadratic compute bottleneck. Qwen2.5-VL combines the two because native-resolution vision needs both: faithful geometry and scalable compute.

### Contribution 2: Dynamic FPS sampling

Qwen2.5-VL extends its dynamic-resolution idea from space into time by introducing **dynamic FPS sampling**. Instead of forcing every video to be processed at one fixed frame rate, the model can consume videos sampled at different FPS values, so different videos — or the same video under different settings — produce temporal token sequences of different lengths. ([arxiv.org](https://arxiv.org/pdf/2502.13923), [huggingface.co](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl))

At a high level, the video pipeline is:

1. sample frames from the raw video at some FPS,
2. resize frames in the same dynamic/native-resolution style used for images,
3. partition the sampled frames into visual patches,
4. group frames temporally before feeding them into the vision encoder,
5. pass timing information forward so the model knows how sampled frames map to real time. ([arxiv.org](https://arxiv.org/pdf/2502.13923), [huggingface.co](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl))

Suppose a video has duration $T$ seconds and is sampled at $f$ frames per second. Then the number of sampled frames is approximately

$$
N_{\text{frames}} \approx T \cdot f
$$

So increasing the FPS increases the number of sampled frames, while decreasing the FPS reduces them.

This is the temporal analogue of dynamic spatial resolution. For images, a larger image produces more spatial tokens. For videos, a higher FPS produces more temporal samples. Qwen2.5-VL is designed to handle this variability rather than assuming one universal sampling rule. ([arxiv.org](https://arxiv.org/pdf/2502.13923))

After sampling, Qwen2.5-VL does not simply treat the video as a flat list of independent images. The paper states that the encoder extends patching into the temporal dimension and groups **two consecutive frames** together. This reduces temporal token count while preserving short-range motion information. ([arxiv.org](https://arxiv.org/pdf/2502.13923))

If the temporal grouping size is 2 frames, then one temporal grid unit corresponds to roughly

$$
\frac{2}{f}
$$

seconds of video.

So for example:

- if $f = 1$, one temporal grid covers about $2$ seconds,
- if $f = 2$, one temporal grid covers about $1$ second,
- if $f = 0.5$, one temporal grid covers about $4$ seconds.

This means that higher FPS gives the model **finer temporal granularity**, while lower FPS gives **coarser but cheaper** video processing.

The Hugging Face pipeline exposes this idea explicitly. When processing video, the processor can take an `fps` argument and also returns timing metadata such as `second_per_grid_ts`, which represents the time interval in seconds for each temporal grid in the 3D position IDs. This is important because it means the model is not only told the order of sampled frames, but also how those samples relate to real elapsed time. ([huggingface.co](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl))

So dynamic FPS sampling is not just “sample more or fewer frames.” It changes both:

1. the **length** of the visual token sequence, and
2. the **temporal resolution** of the representation.

Higher FPS gives more frames, more temporal tokens, and finer timing detail. Lower FPS gives fewer frames and lower compute cost. Qwen2.5-VL is trained to remain robust across this range, which is why the paper describes dynamic FPS sampling as enabling comprehensive video understanding across varied sampling rates. ([arxiv.org](https://arxiv.org/pdf/2502.13923))

This matters because a fixed-FPS pipeline can easily entangle temporal reasoning with one arbitrary sampling convention. For example, “frame 10” means very different elapsed time at $0.5$ FPS and $2$ FPS. Qwen2.5-VL avoids this by pairing dynamic FPS sampling with absolute-time-aware temporal position encoding. In effect, the model learns not just which sampled frame comes next, but how much real time passes between visual observations. ([arxiv.org](https://arxiv.org/pdf/2502.13923))

A concise summary is:

> Dynamic FPS sampling lets Qwen2.5-VL process videos at different frame rates, producing temporal token sequences of varying lengths and granularities. Higher FPS preserves more timing detail, lower FPS saves compute, and the model is explicitly informed how those temporal samples map to real time. ([arxiv.org](https://arxiv.org/pdf/2502.13923), [huggingface.co](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl))
> 
> In simple terms, when the video is long, we sample less FPS, when the video is short, we sample higher FPS.

### The encoder in one sentence

The encoder of Qwen2.5-VL is best understood as a **native-resolution ViT** that replaces absolute position embeddings with **2D RoPE**, replaces most global attention with **window attention**, and then periodically restores **full-image communication** with a few global-attention layers. This is what allows the model to preserve fine spatial structure without paying the full quadratic cost at every layer.
## The merger

In a vision-language model, the vision encoder usually produces a long sequence of visual tokens, while the language model expects tokens that live in its own embedding space and can be processed together with text. The merger, sometimes also called a connector or projector, sits between these two components.

In general, a merger solves two problems:

1. **representation mismatch**: the vision encoder and the LLM use different hidden dimensions and different feature spaces
2. **sequence-length mismatch**: the encoder may output far too many visual tokens for the LLM to process efficiently

A minimal merger only does **dimension alignment**. If the encoder outputs visual tokens

$$
Z = \{z_1, z_2, \dots, z_N\}, \qquad z_i \in \mathbb{R}^{d_v}
$$

then a simple linear projector maps each token into the LLM embedding space:

$$
h_i = W z_i + b, \qquad h_i \in \mathbb{R}^{d_{\text{llm}}}
$$

This solves the shape mismatch, but it does **not** reduce the number of tokens. The LLM still has to process all $N$ visual tokens.

A richer merger may also compress the sequence. In that case, instead of producing $N$ output tokens, it produces a shorter sequence

$$
H = \{h_1, h_2, \dots, h_{N'}\}, \qquad N' < N
$$

so the merger acts as both a projector and a compressor.

### How mergers usually work

There are a few common strategies.

A **linear projector** maps each visual token independently:

$$
h_i = W z_i + b
$$

An **MLP projector** adds a nonlinear transformation:

$$
h_i = W_2 \, \sigma(W_1 z_i + b_1) + b_2
$$

where $\sigma(\cdot)$ is a nonlinearity such as GELU.

A **compression-based merger** first combines neighboring or selected visual tokens into fewer tokens, then projects them into the LLM space.

So the main design question is: should the merger only align dimensions, or should it also reduce token count before the LLM sees the visual sequence?

### What Qwen2.5-VL does

Qwen2.5-VL uses an **MLP-based Vision-Language Merger**. Instead of directly feeding all raw patch features into the language model, it first groups **four spatially adjacent patch features** together.

Suppose the encoder outputs a 2D grid of visual features with spatial size $H \times W$, so the total number of visual tokens is

$$
N = H W
$$

Each patch feature has dimension $d_v$, so each patch token is

$$
z_{i,j} \in \mathbb{R}^{d_v}
$$

Qwen groups each local $2 \times 2$ neighborhood:

$$
\{z_{2a,2b},\; z_{2a,2b+1},\; z_{2a+1,2b},\; z_{2a+1,2b+1}\}
$$

and concatenates them into one larger vector:

$$
g_{a,b} =
\big[
z_{2a,2b};
z_{2a,2b+1};
z_{2a+1,2b};
z_{2a+1,2b+1}
\big]
\in \mathbb{R}^{4d_v}
$$

So after grouping, the spatial grid becomes

$$
\frac{H}{2} \times \frac{W}{2}
$$

and the number of visual tokens becomes

$$
N' = \frac{HW}{4} = \frac{N}{4}
$$

This is the first major effect of the merger: it reduces the number of visual tokens by a factor of 4.

### The actual projection step

After concatenation, the grouped feature is passed through a two-layer MLP to map it into the LLM embedding dimension:

$$
m_{a,b} = W_2 \, \mathrm{GELU}(W_1 g_{a,b} + b_1) + b_2
$$

where

$$
m_{a,b} \in \mathbb{R}^{d_{\text{llm}}}
$$

So the merger is not just reducing token count. It is also translating the local visual summary into the same feature space used by text embeddings.

### What the Hugging Face implementation makes explicit

The Hugging Face implementation shows a slightly more detailed version.

If `spatial_merge_size = 2`, then four neighboring patch features are merged, so the intermediate hidden size is:

$$
d_{\text{merge}} = d_v \cdot 2^2 = 4 d_v
$$

The code applies RMSNorm to the patch features, reshapes the grouped features into a vector of size $4d_v$, and then applies the MLP.

A clean mathematical description is:

$$
\tilde{z}_{i,j} = \mathrm{RMSNorm}(z_{i,j})
$$

$$
\tilde{g}_{a,b} =
\big[
\tilde{z}_{2a,2b};
\tilde{z}_{2a,2b+1};
\tilde{z}_{2a+1,2b};
\tilde{z}_{2a+1,2b+1}
\big]
\in \mathbb{R}^{4d_v}
$$

$$
m_{a,b} = W_2 \, \mathrm{GELU}(W_1 \tilde{g}_{a,b} + b_1) + b_2
$$

So in practical terms, the Qwen merger is:

1. normalize each local patch feature
2. take a $2 \times 2$ spatial neighborhood
3. concatenate the 4 patch vectors
4. apply a two-layer MLP
5. output one LLM-compatible token

### Why this is different from a simple projector

A simple projector would keep the same number of tokens:

$$
N' = N
$$

Qwen’s merger instead gives:

$$
N' = \frac{N}{4}
$$

So it performs **compression and alignment at the same time**. First summarize each small local region, then convert that summary into an LLM token

### Intuition

A good mental model is this:

- the **encoder** produces a dense patch-level representation
- the **merger** summarizes each local $2 \times 2$ patch neighborhood into one token
- the **LLM** reasons over the compressed visual tokens together with text

In other words, the merger behaves like a learned local compression layer.

Instead of sending every single patch token into the LLM, it reduces the token budget while still preserving local structure.

### One-sentence summary

Qwen2.5-VL’s merger is a **local $2 \times 2$ patch compressor plus a two-layer MLP projector**:

$$
\big\{z_{2a,2b}, z_{2a,2b+1}, z_{2a+1,2b}, z_{2a+1,2b+1}\big\}
\;\longrightarrow\;
g_{a,b}
\;\longrightarrow\;
m_{a,b}
\in \mathbb{R}^{d_{\text{llm}}}
$$

which reduces the number of visual tokens by $4\times$ while mapping them into the language model’s embedding space.

## The decoder

In Qwen2.5-VL, the “decoder” is not a separate visual decoder that reconstructs pixels or upsamples features. It is the **Qwen2.5 language-model decoder**: a causal Transformer that takes a single mixed sequence of text tokens and visual tokens, then autoregressively predicts the next text token. The Qwen2.5-VL technical report explicitly describes the model as having three components — a vision encoder, an MLP-based vision-language merger, and a large language model — and says the model is initialized from pretrained **Qwen2.5 LLM** weights, with its original 1D RoPE modified into a multimodal RoPE aligned to absolute time.

So conceptually, the decoder is the part that performs the actual **reasoning and generation**. The encoder and merger prepare visual evidence; the decoder reads that evidence together with the text prompt and decides what to say next. This is why Qwen2.5-VL still feels architecturally closer to a language model than to a classical encoder-decoder vision system.

### What a decoder is in general

A Transformer decoder is a stack of causal self-attention blocks followed by feed-forward blocks. “Causal” means token $t+1$ can only attend to tokens at positions $\le t$, not to future tokens. If the input sequence is

$$
x_1, x_2, \dots, x_T
$$

then the model factorizes generation as

$$
p(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t})
$$

This is the standard autoregressive language-model objective: each token is predicted from the prefix before it. In a multimodal decoder like Qwen2.5-VL, the only twist is that the prefix can contain not just text embeddings, but also visual embeddings inserted into the same sequence. 

A useful contrast is with the original seq2seq Transformer. In that older setup, the decoder has both **self-attention** and **cross-attention** to a separate encoder output. Qwen2.5-VL does not primarily work that way. Instead, after vision features are encoded and merged, they are treated as part of the model’s token stream, and the language model decoder runs over that unified multimodal sequence. The Qwen2.5-VL framework figure and architecture description both present it as a vision encoder feeding a language-model decoder, not as a separate cross-attention-heavy text decoder over frozen encoder memory

### Decoder input: one multimodal token stream

The decoder does not consume raw pixels. By the time information reaches it, the visual side has already produced compressed visual tokens through the encoder and merger. On the text side, the language model uses a token embedding layer:

$$
e_i = \mathrm{Embed}(w_i)
$$

where $w_i$ is a text token id and $e_i \in \mathbb{R}^{d_{\text{llm}}}$ is its embedding.

The visual merger outputs visual tokens in that same language-model hidden space:

$$
v_j \in \mathbb{R}^{d_{\text{llm}}}
$$

So the decoder ultimately processes one interleaved multimodal sequence such as

$$
X = [e_1, e_2, \dots, e_a,\; v_1, v_2, \dots, v_m,\; e_{a+1}, \dots]
$$

This is the key architectural idea: the decoder sees a single sequence in one shared hidden space, rather than separate text and image branches at generation time.

### The Qwen2.5 decoder backbone

The Qwen2.5 technical report says the dense Qwen2.5 models keep a **Transformer-based decoder architecture** and use several standard modern components: **Grouped Query Attention (GQA)** for efficient KV-cache usage, **SwiGLU** as the feed-forward activation, **RoPE** for positional encoding, **QKV bias** in attention, and **RMSNorm with pre-normalization** for stability. Qwen2.5-VL inherits this decoder family, then modifies the positional encoding into multimodal RoPE so that temporal, height, and width information can be represented for visual tokens.

For the released Qwen2.5-VL checkpoints, the decoder sizes are reported directly in the technical report. The 3B model uses hidden size 2048 with 36 layers and 2 KV heads, the 7B model uses hidden size 3584 with 28 layers and 4 KV heads, and the 72B model uses hidden size 8192 with 80 layers and 8 KV heads. The vocabulary size is 151,646 across these checkpoints.

### One decoder layer

The Hugging Face implementation makes the layer structure very explicit. Each decoder layer contains:

1. an input RMSNorm,
2. a self-attention block,
3. a residual connection,
4. a post-attention RMSNorm,
5. an MLP block,
6. another residual connection.

If we denote the layer input by $h^{(\ell)}$, then one layer can be written schematically as:

$$
u^{(\ell)} = \mathrm{RMSNorm}(h^{(\ell)})
$$

$$
a^{(\ell)} = \mathrm{SelfAttention}(u^{(\ell)})
$$

$$
\tilde{h}^{(\ell)} = h^{(\ell)} + a^{(\ell)}
$$

$$
r^{(\ell)} = \mathrm{RMSNorm}(\tilde{h}^{(\ell)})
$$

$$
m^{(\ell)} = \mathrm{MLP}(r^{(\ell)})
$$

$$
h^{(\ell+1)} = \tilde{h}^{(\ell)} + m^{(\ell)}
$$

This is the classic pre-norm residual Transformer pattern: normalize, transform, add back the skip path, then repeat for the feed-forward block.

### Self-attention inside the decoder

Inside the attention block, queries, keys, and values are computed from the current hidden states. Abstractly:

$$
Q = h W_Q,\qquad K = h W_K,\qquad V = h W_V
$$

and causal self-attention is

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M_{\text{causal}}\right)V
$$

where $M_{\text{causal}}$ masks out future positions. The implementation builds these masks before the decoder stack, then passes the appropriate mask to each layer. The same code path also supports cached keys and values for fast autoregressive decoding, which is exactly what makes token-by-token generation efficient at inference time.

Grouped Query Attention matters here because it reduces the KV-cache burden. Instead of allocating separate key/value heads for every query head, several query heads share the same key/value heads. That preserves most of the benefits of multi-head attention while making long-context generation cheaper in memory and usually faster in practice. Qwen2.5 explicitly lists GQA as one of the core decoder design choices.

### The MLP block

The feed-forward sublayer in the decoder is not a plain ReLU MLP. In the implementation, the decoder MLP uses the gated form associated with SwiGLU-style designs:

$$
\mathrm{MLP}(x) = W_{\text{down}}\big(\phi(W_{\text{gate}}x) \odot W_{\text{up}}x\big)
$$

where $\phi$ is the activation function and $\odot$ is elementwise multiplication. In Qwen2.5, the technical report states that SwiGLU is the non-linear activation used in the decoder family, and the implementation exposes the gated structure directly through separate `gate_proj`, `up_proj`, and `down_proj` linear maps.

This matters because the MLP is not just a cheap post-processing block. In large language models, a substantial amount of representation transformation happens inside the feed-forward path. The attention block mixes information across positions; the MLP block remaps and enriches that information at each position.

### Contribution 3: Positional encoding in the decoder: from RoPE to MRoPE

A normal language-model decoder uses **1D RoPE** because text is naturally a one-dimensional sequence: token 1, token 2, token 3, and so on. In that setting, each token only needs one position index. But visual tokens are not naturally one-dimensional. An image token lives on a **2D grid** with height and width coordinates, and a video token lives on a **3D spatiotemporal grid** with temporal, height, and width coordinates. If we flatten all of that into one long 1D sequence and only use ordinary RoPE, the model loses important structural information about where a token came from in space and time. That is the motivation for **MRoPE**: it extends RoPE so the model can represent **temporal position, vertical position, and horizontal position separately**. 

In Qwen2-VL, MRoPE decomposes position into three components:

- **temporal**
- **height**
- **width**

For text tokens, all three indices are the same, so MRoPE reduces back to ordinary 1D RoPE behavior. For image tokens, the temporal index is constant while height and width vary with spatial location. For video tokens, all three can vary: the temporal index changes across frames, while height and width change across patch locations inside each frame. The Hugging Face implementation states this explicitly: for vision embeddings, rotary position embedding is applied separately on the **temporal, height, and width** dimensions, while for text embeddings the three indices are identical, so text behaves just like a normal modern LLM. 

A compact way to think about this is:

- for text tokens:
  $$
  (p,\; p,\; p)
  $$
- for image tokens:
  $$
  (0,\; h,\; w)
  $$
- for video tokens:
  $$
  (t,\; h,\; w)
  $$

So MRoPE is really saying: the decoder should not only know **where a token is in the sequence**, but also **where it is in space and time**. This is especially important for images, documents, charts, and videos, where meaning depends heavily on layout and timing rather than just left-to-right order.

#### Why plain 1D RoPE is not enough

Ordinary 1D RoPE works well for text because text is already serialized in a meaningful order. But multimodal inputs are different. Suppose two image patches are adjacent vertically in the original image. After flattening, they may end up far apart in the 1D token sequence depending on the scan order. Conversely, two patches that are neighbors in the flattened sequence may not be meaningful neighbors in the image. The same problem becomes even worse for video, where the model should ideally know not only “which token comes next,” but also “which frame this token belongs to” and “how much time separates it from other frames.” MRoPE is needed because it preserves this structure inside the decoder’s attention mechanism instead of forcing the model to infer it from a lossy flattened order.

Another practical benefit noted in the Qwen2-VL paper is that MRoPE keeps the image and video position IDs smaller and more structured than a naive 1D flattening scheme, which helps the model extrapolate to longer multimodal sequences at inference time. In other words, MRoPE is not only more faithful to multimodal geometry; it is also a better positional parameterization for long-context multimodal decoding.

#### What MRoPE changes mathematically

MRoPE does **not** replace self-attention with a new attention mechanism. The decoder is still a causal Transformer. What changes is how rotary position encoding is applied to the query and key vectors.

With standard RoPE, a token at position $p$ is rotated using one 1D position index. With MRoPE, the model splits the channel dimensions into three sections and applies rotary embedding separately for:

- temporal position
- height position
- width position

The Hugging Face implementation literally documents this split: the channel dimension is divided into three chunks for temporal, height, and width rotary embedding, and then recombined before attention is computed. So the attention block is still self-attention, but the notion of “position” inside that attention is now **multimodal and structured**, not purely 1D.

A schematic way to write the idea is:

$$
q' = \mathrm{RoPE}_t(q_t)\; \oplus \; \mathrm{RoPE}_h(q_h)\; \oplus \; \mathrm{RoPE}_w(q_w)
$$

$$
k' = \mathrm{RoPE}_t(k_t)\; \oplus \; \mathrm{RoPE}_h(k_h)\; \oplus \; \mathrm{RoPE}_w(k_w)
$$

where the query and key channels are partitioned into temporal, height, and width sections. The exact implementation interleaves these chunks rather than using this exact notation, but conceptually this is what is happening: different parts of the embedding encode different positional axes.

#### Why Qwen2.5-VL upgrades MRoPE with absolute time

Qwen2-VL already had MRoPE, but its temporal IDs were tied to the **number of input frames**. The Qwen2.5-VL technical report identifies the limitation directly: frame-count-based temporal IDs do not account for **the speed of content changes** or the **absolute timing of events** in the original video. If two videos are sampled at different FPS values, the same frame index may correspond to very different amounts of elapsed time. That makes frame index an unstable notion of time.

Qwen2.5-VL fixes this by aligning the **temporal component of MRoPE with absolute time**. Instead of only encoding frame order, the temporal IDs are aligned with timestamps, so the intervals between temporal IDs reflect real elapsed time. The paper says this lets the model learn the “tempo of time” through those intervals and achieve more consistent temporal alignment across videos sampled at different FPS rates, without adding any extra temporal heads or extra computational overhead.

This matters because video understanding often depends on **duration**, not just order. For example, “the person immediately sat down after standing up” and “the person sat down ten minutes later” have the same ordering but very different temporal meaning. Likewise, in temporal grounding, the model needs to know not just which frame came before which, but where an event occurs in real time. Absolute-time MRoPE helps the model encode exactly that distinction.

#### How MRoPE interacts with dynamic FPS sampling

Dynamic FPS sampling changes how densely a video is sampled, which means the model may see the same underlying event through different numbers of frames depending on the sampling rate. Without absolute-time-aware positional encoding, this would make temporal reasoning brittle: the decoder might learn frame-number patterns that do not transfer across FPS settings. Qwen2.5-VL’s contribution is that dynamic FPS sampling and absolute-time MRoPE work together: the former changes the temporal sampling density, and the latter ensures that temporal position still tracks **real elapsed time** rather than just sampled-frame order. That pairing is one reason the paper highlights both as linked contributions for long-video understanding and temporal localization.

#### The real role of MRoPE

The most important point is that MRoPE is not a minor implementation detail. It is the mechanism that lets the decoder remain a standard causal language model while still being aware of **space** and **time**. Without it, the LLM would only see a flat sequence of multimodal tokens and would have to recover geometry and timing indirectly. With MRoPE, those structural cues are built directly into the attention computation. That is why MRoPE is one of the key reasons Qwen2.5-VL is more than “just a text LLM after an image encoder.”

### Causal masking and generation

Because the decoder is autoregressive, it uses causal masking during generation. The implementation constructs the causal mask, computes the rotary position embeddings, then iterates through the decoder layers. If KV-cache is enabled, the past key/value states are reused so that the model does not recompute the whole prefix for every new output token. That is why generation scales much better than naive re-forwarding.

At a high level, the generation loop is:

1. build the multimodal prefix,
2. run it through the decoder,
3. produce logits for the next token,
4. sample or select the next token,
5. append it to the sequence,
6. reuse cached KV states for the next step.

So even though the model may have started from images or video, the output side is still standard autoregressive language generation. The Hugging Face docs demonstrate this through `Qwen2_5_VLForConditionalGeneration` and `generate()`, which expose the model exactly as a text-generating multimodal causal LM.

### Why there is no separate cross-attention decoder

This is an easy place to get confused. Many multimodal architectures use a text decoder that cross-attends to a frozen or separate encoder memory. Qwen2.5-VL instead pushes most of the modality alignment earlier: the encoder extracts visual features, the merger compresses and projects them into the LLM hidden space, and the decoder operates over a unified sequence. So the decoder remains very close to a standard LLM decoder stack. The main multimodal changes are in the token stream and the positional encoding, not in the addition of a heavy cross-attention subnetwork at every layer.

### Final output layer

After the stack of decoder layers, the hidden states pass through a final RMSNorm. In a causal language model, those final hidden states are then mapped into vocabulary logits, and the next token is predicted from that distribution. Even when the prompt contains image or video content, the model’s outward behavior is still: “produce the next text token.” That is why the most faithful mental model of Qwen2.5-VL’s decoder is simply **a Qwen2.5 causal LM that has been taught to consume multimodal tokens**.

### One-sentence summary

Qwen2.5-VL’s decoder is a **pretrained Qwen2.5 causal Transformer decoder** with **GQA, RMSNorm, and SwiGLU**, upgraded with **multimodal RoPE** so it can process text tokens and merged visual tokens in one autoregressive sequence and generate the response token by token. 


## How inference works

At inference time, Qwen2.5-VL follows a fairly clean path from pixels to text. The processor first prepares the visual input by resizing it into the native-resolution tokenization scheme, sampling video frames at the chosen FPS when needed, and generating the multimodal position metadata. The vision encoder turns those inputs into visual features, the merger compresses and projects them into the language-model space, and the decoder then autoregressively generates text from one mixed sequence of text tokens and visual tokens.

In practical terms, the loop looks like this:

1. preprocess images or videos into visual tensors and temporal metadata
2. encode them with the ViT-based vision encoder
3. compress and project them with the merger
4. concatenate the merged visual tokens with the text prompt
5. decode the response token by token with the Qwen2.5 language model

That pipeline is conceptually simple, but the quality of the output depends heavily on the earlier architectural choices discussed above. Better spatial fidelity, better temporal alignment, and better token compression all directly affect what the decoder is able to reason over.

## Sources

- [Qwen2.5-VL Technical Report, submitted February 19, 2025](https://arxiv.org/abs/2502.13923)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864?utm_source=chatgpt.com)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030?utm_source=chatgpt.com)
- [Rotary Position Embedding for Vision Transformer](https://arxiv.org/abs/2403.13298?utm_source=chatgpt.com)
- [A Survey of Connectors in Multi-modal Large Language Models](https://arxiv.org/abs/2502.11453?utm_source=chatgpt.com)
- [LLaVA-1.5 / Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744?utm_source=chatgpt.com)
- <a href="https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_5_vl">Hugging Face Transformers documentation for Qwen2.5-VL</a>
