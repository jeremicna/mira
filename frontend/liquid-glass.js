// Liquid Glass Effect (Modified to use #glasselement)
// Original by Shu Ding — modified to attach to an existing element

(function () {
	"use strict";

	// Destroy previous version
	if (window.liquidGlass) {
		window.liquidGlass.destroy();
	}

	// Utility functions
	function smoothStep(a, b, t) {
		t = Math.max(0, Math.min(1, (t - a) / (b - a)));
		return t * t * (3 - 2 * t);
	}

	function length(x, y) {
		return Math.sqrt(x * x + y * y);
	}

	function roundedRectSDF(x, y, width, height, radius) {
		const qx = Math.abs(x) - width + radius;
		const qy = Math.abs(y) - height + radius;
		return Math.min(Math.max(qx, qy), 0) + length(Math.max(qx, 0), Math.max(qy, 0)) - radius;
	}

	function texture(x, y) {
		return { type: "t", x, y };
	}

	function generateId() {
		return "liquid-glass-" + Math.random().toString(36).substr(2, 9);
	}

	class Shader {
		constructor(options = {}) {
			this.width = options.width || 400;
			this.height = options.height || 200;
			this.fragment = options.fragment;
			this.canvasDPI = 1;
			this.id = generateId();

			this.mouse = { x: 0, y: 0 };
			this.mouseUsed = false;

			this.createElement();
			this.setupEventListeners();
			this.updateShader();
		}

		createElement() {
			this.container = document.getElementById("glasselement");
			if (!this.container) throw new Error("Element #glasselement not found.");

			Object.assign(this.container.style, {
				position: "absolute",
				width: this.width + "px",
				height: this.height + "px",
				overflow: "hidden",
				borderRadius: "40px",
				cursor: "grab",
				zIndex: 9999,
				pointerEvents: "auto",
				backdropFilter: `url(#${this.id}_filter) blur(4px) contrast(1.0) brightness(1.05) saturate(1.1)`,
				boxShadow: `
					0 8px 16px rgba(0, 0, 0, 0.25),
					inset 0 -10px 25px rgba(0, 0, 0, 0.15),
					inset 0 -3px 8px rgba(255, 255, 255, 0.7),
					inset 0 -1.5px 3px rgba(255, 255, 255, 0.4),
					inset -1.5px 0 3px rgba(255, 255, 255, 0.4),
					inset 1.5px 0 3px rgba(0, 0, 0, 0.1),
					inset 0 1.5px 3px rgba(0, 0, 0, 0.1)
				`,
			});

			this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
			this.svg.setAttribute("width", "0");
			this.svg.setAttribute("height", "0");
			this.svg.style.position = "fixed";
			this.svg.style.top = "0";
			this.svg.style.left = "0";
			this.svg.style.zIndex = "9998";

			const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
			const filter = document.createElementNS("http://www.w3.org/2000/svg", "filter");

			filter.setAttribute("id", `${this.id}_filter`);
			filter.setAttribute("filterUnits", "userSpaceOnUse");
			filter.setAttribute("colorInterpolationFilters", "sRGB");
			filter.setAttribute("width", this.width);
			filter.setAttribute("height", this.height);

			this.feImage = document.createElementNS("http://www.w3.org/2000/svg", "feImage");
			this.feImage.setAttribute("id", `${this.id}_map`);
			this.feImage.setAttribute("width", this.width);
			this.feImage.setAttribute("height", this.height);

			this.feDisplacementMap = document.createElementNS(
				"http://www.w3.org/2000/svg",
				"feDisplacementMap"
			);
			this.feDisplacementMap.setAttribute("in", "SourceGraphic");
			this.feDisplacementMap.setAttribute("in2", `${this.id}_map`);
			this.feDisplacementMap.setAttribute("xChannelSelector", "R");
			this.feDisplacementMap.setAttribute("yChannelSelector", "G");

			filter.appendChild(this.feImage);
			filter.appendChild(this.feDisplacementMap);
			defs.appendChild(filter);
			this.svg.appendChild(defs);

			this.canvas = document.createElement("canvas");
			this.canvas.width = this.width * this.canvasDPI;
			this.canvas.height = this.height * this.canvasDPI;
			this.canvas.style.display = "none";

			this.context = this.canvas.getContext("2d");
		}

		setupEventListeners() {
			let dragging = false;
			let startX, startY, initialX, initialY;

			this.container.addEventListener("mousedown", (e) => {
				dragging = true;
				this.container.style.cursor = "grabbing";
				startX = e.clientX;
				startY = e.clientY;
				const rect = this.container.getBoundingClientRect();
				initialX = rect.left;
				initialY = rect.top;
				e.preventDefault();
			});

			document.addEventListener("mousemove", (e) => {
				if (dragging) {
					const dx = e.clientX - startX;
					const dy = e.clientY - startY;
					this.container.style.left = initialX + dx + "px";
					this.container.style.top = initialY + dy + "px";
				}

				const rect = this.container.getBoundingClientRect();
				this.mouse.x = (e.clientX - rect.left) / rect.width;
				this.mouse.y = (e.clientY - rect.top) / rect.height;

				if (this.mouseUsed) this.updateShader();
			});

			document.addEventListener("mouseup", () => {
				dragging = false;
				this.container.style.cursor = "grab";
			});
		}

		updateShader() {
			const mouseProxy = new Proxy(this.mouse, {
				get: (t, p) => {
					this.mouseUsed = true;
					return t[p];
				},
			});

			this.mouseUsed = false;

			const w = this.width * this.canvasDPI;
			const h = this.height * this.canvasDPI;

			const data = new Uint8ClampedArray(w * h * 4);
			const raw = [];
			let max = 0;

			for (let i = 0; i < data.length; i += 4) {
				const x = (i / 4) % w;
				const y = Math.floor(i / 4 / w);

				const out = this.fragment({ x: x / w, y: y / h }, mouseProxy);
				const dx = out.x * w - x;
				const dy = out.y * h - y;

				raw.push(dx, dy);
				max = Math.max(max, Math.abs(dx), Math.abs(dy));
			}

			max *= 0.5;

			let idx = 0;
			for (let i = 0; i < data.length; i += 4) {
				data[i] = (raw[idx++] / max + 0.5) * 255;
				data[i + 1] = (raw[idx++] / max + 0.5) * 255;
				data[i + 2] = 0;
				data[i + 3] = 255;
			}

			this.context.putImageData(new ImageData(data, w, h), 0, 0);
			this.feImage.setAttributeNS(
				"http://www.w3.org/1999/xlink",
				"href",
				this.canvas.toDataURL()
			);
			this.feDisplacementMap.setAttribute("scale", max.toString());
		}

		appendTo(parent) {
			parent.appendChild(this.svg);
			parent.appendChild(this.canvas);
		}

		destroy() {
			this.svg.remove();
			this.canvas.remove();
		}
	}

	function createLiquidGlass() {
		const shader = new Shader({
			width: 400,
			height: 600,
			fragment: (uv, mouse) => {
			  	const ix = uv.x - 0.5;
				const iy = uv.y - 0.5;
				const dist = roundedRectSDF(ix, iy, 0.3, 0.2, 0);
				const disp = smoothStep(0.4, 0, dist - 0.16);
				const scaled = smoothStep(0, 1, disp);
				return texture(ix * scaled + 0.5, iy * scaled + 0.5);
			},
		});

		shader.appendTo(document.body);
		window.liquidGlass = shader;
	}

	createLiquidGlass();
})();
