import 'katex/dist/katex.min.css';
import React from 'react';
import { BlockMath, InlineMath } from 'react-katex';

export const KatexBlock = ({ math, inline = false }) => {
  if (inline) {
    return <InlineMath math={math} />;
  }
  return <BlockMath math={math} />;
};

export const KatexInline = ({ math }) => {
  return <InlineMath math={math} />;
}; 