import streamlit as st
from PIL import Image



import streamlit as st

def custom_animation():
    """使用 HTML 和 CSS 创建自定义动画效果"""
    animation_code = """
    <div id="custom-animation" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 1000;">
        <!-- 动画内容将通过 JavaScript 添加 -->
    </div>
    
    <script>
    // 创建自定义动画效果
    function createCustomAnimation() {
        const container = document.getElementById('custom-animation');
        container.innerHTML = '';  // 清空容器
        
        // 创建10个彩色元素
        for (let i = 0; i < 10; i++) {
            const element = document.createElement('div');
            const size = Math.random() * 30 + 20;  // 随机大小
            const color = `rgb(${Math.random() * 200 + 55}, ${Math.random() * 200 + 55}, ${Math.random() * 200 + 55})`;
            
            element.style.position = 'absolute';
            element.style.bottom = '-50px';
            element.style.left = `${Math.random() * 100}%`;
            element.style.width = `${size}px`;
            element.style.height = `${size}px`;
            element.style.borderRadius = '50%';
            element.style.backgroundColor = color;
            element.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
            element.style.animation = `float ${Math.random() * 5 + 5}s linear infinite`;
            
            container.appendChild(element);
        }
        
        // 添加CSS动画
        const style = document.createElement('style');
        style.textContent = `
            @keyframes float {
                0% {
                    transform: translateY(0) rotate(0deg);
                    opacity: 1;
                }
                100% {
                    transform: translateY(-100vh) rotate(360deg);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    // 页面加载时创建动画
    createCustomAnimation();
    
    // 可以添加按钮事件监听来触发动画
    </script>
    """
    
    st.markdown(animation_code, unsafe_allow_html=True)

# 在应用中使用自定义动画
if st.button("显示自定义动画"):
    custom_animation()

# st.title('YES')

# if st.button("点击庆祝"):
#     st.balloons()
#     st.success("嘿嘿")




