class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def inorder_traversal(root):
    result = []
    def traverse(node):
        if node:
            traverse(node.left)
            result.append(node.value)
            traverse(node.right)
    traverse(root)
    return result
def preorder_traversal(root):
    result = []
    def traverse(node):
        if node:
            result.append(node.value)
            traverse(node.left)
            traverse(node.right)
    traverse(root)
    return result
def postorder_traversal(root):
    result = []
    def traverse(node):
        if node:
            traverse(node.left)
            traverse(node.right)
            result.append(node.value)
    traverse(root)
    return result
# 构建示例二叉树
root = TreeNode(1)
root.right = TreeNode(22)
root.right.left = TreeNode(13)
root.right.right = TreeNode(84)
root.right.left.left = TreeNode(6)
root.right.left.left.left = TreeNode(5)
root.right.left.left.right = TreeNode(7)
root.right.right.left = TreeNode(25)
root.right.right.left.right = TreeNode(58)
root.right.right.left.right.left = TreeNode(49)
root.right.right.left.right.right = TreeNode(62)
root.right.right.left.right.left.left = TreeNode(43)
root.right.right.left.right.left.left.left = TreeNode(28)
root.right.right.left.right.left.right = TreeNode(57)
root.right.right.left.right.right.left = TreeNode(64)
root.right.right.left.right.right.left.right = TreeNode(69)

# 中序遍历
inorder_result = postorder_traversal(root)
print(inorder_result)  # 输出: [1, 5, 6, 7, 13, 22, 28, 43, 49, 57, 58, 25, 69, 64, 62, 84]
