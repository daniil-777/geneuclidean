/// <reference types="vite/client" />

declare module '3dmol' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const _3Dmol: any;
  export default _3Dmol;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const SurfaceType: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export function createViewer(element: HTMLElement, config?: Record<string, unknown>): any;
}
